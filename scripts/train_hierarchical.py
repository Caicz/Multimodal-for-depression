import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel
from torch.optim import AdamW
from torchvision import transforms
from PIL import Image
import json
import random
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from hierarchical_model import HierarchicalMultimodalModel
from torch.cuda.amp import autocast, GradScaler

# --- 基础配置 ---
BATCH_SIZE = 8    # 适度增加，提升 GPU 饱和度
GRAD_ACCUMULATION = 2 # 维持等效 Batch Size = 64
MAX_SEQ_LEN = 25
TEXT_MAX_LEN = 128
LR = 2e-4

class UserSequenceDataset(Dataset):
    def __init__(self, sequences, tokenizer, transform=None, use_vlm=False, vlm_tokenizer=None):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.vlm_tokenizer = vlm_tokenizer or tokenizer
        self.use_vlm = use_vlm
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_data = self.sequences[idx]
        label = seq_data['label']
        tweets = seq_data['sequence']
        project_root = Path(__file__).parent.parent.absolute()
        
        input_ids_list, attn_mask_list, pixels_list, seq_mask, rt_list = [], [], [], [], []
        
        # VLM 总结文本处理 (用户级)
        if self.use_vlm and 'vlm_summary' in seq_data:
            s_enc = self.vlm_tokenizer(seq_data['vlm_summary'], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
            sum_ids = s_enc['input_ids'].squeeze(0)
            sum_mask = s_enc['attention_mask'].squeeze(0)
        else:
            sum_ids = torch.zeros(128, dtype=torch.long)
            sum_mask = torch.zeros(128, dtype=torch.long)

        for i in range(MAX_SEQ_LEN):
            if i < len(tweets):
                t = tweets[i]
                # 原始推文文本
                enc = self.tokenizer(t['text'], padding='max_length', truncation=True, max_length=TEXT_MAX_LEN, return_tensors='pt')
                input_ids_list.append(enc['input_ids'].squeeze(0))
                attn_mask_list.append(enc['attention_mask'].squeeze(0))
                rt_list.append(1 if t.get('is_retweet', False) else 0)

                if t.get('has_image', False):
                    img_path = project_root / t['user_path'] / f"{t['tweet_id']}.jpg"
                    try:
                        img = Image.open(img_path).convert('RGB')
                        pixels_list.append(self.transform(img))
                    except:
                        pixels_list.append(torch.zeros(3, 224, 224))
                        #print("图片加载失败，路径:", img_path)
                else:
                    pixels_list.append(torch.zeros(3, 224, 224))
                seq_mask.append(True)
            else:
                input_ids_list.append(torch.zeros(TEXT_MAX_LEN, dtype=torch.long))
                attn_mask_list.append(torch.zeros(TEXT_MAX_LEN, dtype=torch.long))
                pixels_list.append(torch.zeros(3, 224, 224))
                seq_mask.append(False)
                rt_list.append(0)

        return {
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attn_mask_list),
            'sum_ids': sum_ids,
            'sum_mask': sum_mask,
            'pixel_values': torch.stack(pixels_list),
            'seq_mask': torch.tensor(seq_mask),
            'risk_density': torch.tensor(seq_data.get('risk_density', 0.0), dtype=torch.float),
            'is_retweet': torch.tensor(rt_list, dtype=torch.long),
            'label': torch.tensor(label)
        }

def evaluate_on_test(model, test_data, tokenizer, device, report_path, use_vlm=False, vlm_tokenizer=None):
    test_ds = UserSequenceDataset(test_data, tokenizer, use_vlm=use_vlm, vlm_tokenizer=vlm_tokenizer)
    test_loader = DataLoader(
        test_ds, 
        batch_size=8, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids, mask, pixels, s_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['pixel_values'].to(device), batch['seq_mask'].to(device)
            sum_ids, sum_mask = batch['sum_ids'].to(device), batch['sum_mask'].to(device)
            densities, rts, labels = batch['risk_density'].to(device), batch['is_retweet'].to(device), batch['label'].to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float32):
                outputs = model(ids, mask, pixels, s_mask, densities, rts, 
                                summary_ids=sum_ids if use_vlm else None, 
                                summary_mask=sum_mask if use_vlm else None)
                preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    report = classification_report(all_labels, all_preds, target_names=['Negative', 'Positive'], digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    output_str = f"\n==================================================\n测试集评估报告\n==================================================\n{report}\n混淆矩阵:\n{cm}\n"
    print(output_str)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f: f.write(output_str)

def train_single_ablation(ablation, dataset, train_data, val_data, test_data, tokenizer, device, epochs=15):
    """训练单个消融变体，返回最佳验证准确率"""
    print(f"\n{'='*60}\n开始训练消融变体: {ablation}\n{'='*60}")
    
    # 选择预训练模型路径（与原始保持一致）
    if dataset == 'swdd':
        model_id = 'hfl/chinese-roberta-wwm-ext'
    else:
        model_id = 'bert-base-uncased'
    local_model_path = Path(f"../models/{model_id}")
    bert_load_path = str(local_model_path) if local_model_path.exists() else model_id

    # 创建模型，传入 ablation 参数
    model = HierarchicalMultimodalModel(
        num_classes=2,
        seq_len=MAX_SEQ_LEN,
        bert_model_name=bert_load_path,
        ablation=ablation
    ).to(device)

    train_loader = DataLoader(
        UserSequenceDataset(train_data, tokenizer, use_vlm=False),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        UserSequenceDataset(val_data, tokenizer, use_vlm=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # 优化器：仅优化 requires_grad=True 的参数（已冻结主干网络）
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    best_acc = 0
    patience, no_improve = 6, 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            ids, mask, pixels, s_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['pixel_values'].to(device), batch['seq_mask'].to(device)
            sum_ids, sum_mask = batch['sum_ids'].to(device), batch['sum_mask'].to(device)
            densities, rts, labels = batch['risk_density'].to(device), batch['is_retweet'].to(device), batch['label'].to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float32):
                # 消融实验中不使用 VLM，传入 None
                outputs = model(ids, mask, pixels, s_mask, densities, rts,
                                summary_ids=None, summary_mask=None)
                loss = criterion(outputs, labels) / GRAD_ACCUMULATION

            scaler.scale(loss).backward()
            if (i + 1) % GRAD_ACCUMULATION == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss += loss.item() * GRAD_ACCUMULATION
            if (i + 1) % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item() * GRAD_ACCUMULATION:.4f}")

        scheduler.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                ids, mask, pixels, s_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['pixel_values'].to(device), batch['seq_mask'].to(device)
                sum_ids, sum_mask = batch['sum_ids'].to(device), batch['sum_mask'].to(device)
                densities, rts, labels = batch['risk_density'].to(device), batch['is_retweet'].to(device), batch['label'].to(device)
                with torch.amp.autocast(device_type=device.type, dtype=torch.float32):
                    outputs = model(ids, mask, pixels, s_mask, densities, rts,
                                    summary_ids=None, summary_mask=None)
                    correct += (outputs.argmax(dim=1) == labels).sum().item()

        acc = correct / len(val_data)
        print(f"Epoch {epoch+1}, Val Acc: {acc:.4f}, Avg Loss: {total_loss/len(train_loader):.4f}")
        if acc > best_acc:
            best_acc, no_improve = acc, 0
            # 修改：保存模型和报告时加入 ablation 后缀
            model_path = f"../models/best_model_{dataset}_{ablation}.pt"
            torch.save(model.state_dict(), model_path)
            report_path = f"../results/report_{dataset}_{ablation}.txt"
            evaluate_on_test(model, test_data, tokenizer, device, report_path, use_vlm=False)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"早停于 epoch {epoch+1}")
                break
    return best_acc

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='swdd', choices=['swdd', 'twitter'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--use_vlm', action='store_true', help='是否加载 VLM 总结特征')
    parser.add_argument('--ablation', type=str, default='full', help='消融实验类型，支持逗号分隔多个，如 full,img,textonly')
    parser.add_argument('--debug', action='store_true', help='启用调试模式，使用更小的数据集和更少的训练轮数')
    args = parser.parse_args()
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 动态构建数据路径
    data_dir = Path("../data")
    if args.ablation == 'fliter':
        file_name = f"user_sequences_{args.dataset}_ablation.jsonl"
    else:
        file_name = f"user_sequences_{args.dataset}.jsonl"
    data_path = data_dir / file_name
    
    if not data_path.exists():
        print(f"错误: 找不到文件 {data_path}")
        return

    # 针对不同数据集选择合适的模型
    model_id = 'hfl/chinese-roberta-wwm-ext' if args.dataset == 'swdd' else 'bert-base-uncased'
    
    # 检查本地是否有下载好的模型
    local_model_path = Path(f"../models/{model_id}")
    bert_load_path = str(local_model_path) if local_model_path.exists() else model_id

    print(f"正在加载数据集: {args.dataset}, 使用模型路径/ID: {bert_load_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]
    if args.use_vlm and data_test_path.exists():
        with open(data_test_path, 'r', encoding='utf-8') as f:
            test_all_data = [json.loads(line) for line in f]

    if args.debug:
        _,all_data = train_test_split(all_data, train_size=1.0, random_state=42)
    else:    
        train_val_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
        train_data_full, val_data_full = train_test_split(train_val_data, test_size=0.125, random_state=42)
    
    if args.ablation != 'none':
        train_data, _ = train_test_split(
            train_data_full, train_size=0.1, random_state=42,
            stratify=[d['label'] for d in train_data_full]
        )
        val_data, _ = train_test_split(
            val_data_full, train_size=0.1, random_state=42,
            stratify=[d['label'] for d in val_data_full]
        )
    else:
        train_data, val_data = train_data_full, val_data_full

    # 加载 Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(bert_load_path)
        vlm_tokenizer = None
    except Exception as e:
        print(f"加载模型失败！请确保已将模型文件上传到 {local_model_path}")
        print(f"错误信息: {e}")
        return

    ablation_list = [a.strip() for a in args.ablation.split(',')]

    train_loader = DataLoader(
        UserSequenceDataset(train_data, tokenizer, use_vlm=args.use_vlm, vlm_tokenizer=vlm_tokenizer), 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,        # 开启 8 个子进程并行读图和预处理
        pin_memory=True       # 开启内存锁定，加速数据传输到 GPU
    )
    val_loader = DataLoader(
        UserSequenceDataset(val_data, tokenizer, use_vlm=args.use_vlm, vlm_tokenizer=vlm_tokenizer), 
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True
    )
    
    model = HierarchicalMultimodalModel(num_classes=2, seq_len=MAX_SEQ_LEN, unfreeze_bert=True, unfreeze_resnet=True, bert_model_name=bert_load_path).to(device)
    
    # 使用 2 块 GPU 进行 DataParallel
    if torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 块 GPU，启用 DataParallel...")
        model = nn.DataParallel(model)

    # 获取原始模型引用以方便访问内部组件
    raw_model = model.module if hasattr(model, 'module') else model

    optimizer = AdamW([
        {'params': raw_model.bert.parameters(), 'lr': 1e-5}, 
        {'params': raw_model.resnet_backbone.parameters(), 'lr': 1e-5},
        {'params': raw_model.vlm_bert.parameters(), 'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters() if 'bert' not in n and 'resnet' not in n], 'lr': LR}
    ], weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')  # 混合精度训练的梯度缩放器
    best_acc = 0
    patience, no_improve = 6, 0
    
    print(f"配置完毕: use_vlm={args.use_vlm}, Device: {device}")

    results = {}
    for abla in ablation_list:
        best_acc = train_single_ablation(
            ablation=abla,
            dataset=args.dataset,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            tokenizer=tokenizer,
            device=device,
            epochs=args.epochs
        )
        results[abla] = best_acc
        print(f"消融变体 {abla} 最佳验证准确率: {best_acc:.4f}")

    print("\n所有消融实验完成！汇总结果：")
    for abla, acc in results.items():
        print(f"  {abla}: {acc:.4f}")

if __name__ == "__main__":
    train()
