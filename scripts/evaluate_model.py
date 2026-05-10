import torch
import json
import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 添加脚本目录以便导入
sys.path.append(str(Path(__file__).parent))
from hierarchical_model import HierarchicalMultimodalModel
from train_hierarchical import UserSequenceDataset

# --- 配置 ---
DATA_PATH = r"E:\毕设数据集\multimodal_work\data\user_sequences.jsonl"
MODEL_PATH = r"E:\毕设数据集\multimodal_work\models\best_multimodal_model.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4 

def evaluate():
    print(f"正在准备评估环境 (设备: {DEVICE})...")
    
    # 1. 加载并划分验证集
    if not os.path.exists(DATA_PATH):
        print("错误：找不到数据文件")
        return
        
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]
    
    # 7:1:2 划分 (获取 20% 的测试集进行最终评估)
    _, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    test_ds = UserSequenceDataset(test_data, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 加载模型
    model = HierarchicalMultimodalModel(num_classes=2, seq_len=25, unfreeze_bert=True, unfreeze_resnet=True, bert_model_name='hfl/chinese-roberta-wwm-ext').to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    print(f"开始对 {len(test_data)} 个样本进行评估...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            pixels = batch['pixel_values'].to(DEVICE)
            s_mask = batch['seq_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            # 使用半精度计算以提速
            from torch.cuda.amp import autocast
            with autocast(enabled=(DEVICE.type == 'cuda')):
                outputs = model(ids, mask, pixels, s_mask)
                preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if (i+1) % 20 == 0:
                print(f"  已处理 {min((i+1)*BATCH_SIZE, len(test_data))}/{len(test_data)} 个用户...")

    # 3. 计算指标
    report = classification_report(all_labels, all_preds, target_names=['Negative', 'Positive'], digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    
    output_str = f"""==================================================
           模型评估报告 (Classification Report)
==================================================
{report}

混淆矩阵 (Confusion Matrix):
               预测 Negative   预测 Positive
实际 Negative    {cm[0][0]:<15} {cm[0][1]}
实际 Positive    {cm[1][0]:<15} {cm[1][1]}
=================================================="""

    print(output_str)
    
    # 保存到文件
    results_path = r"E:\毕设数据集\multimodal_work\results\evaluation_report.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(output_str)
    print(f"\n评估报告已保存至: {results_path}")

if __name__ == "__main__":
    evaluate()
