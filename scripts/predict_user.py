import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer
from torchvision import transforms
from PIL import Image
import json
import os
import sys
from pathlib import Path

# 添加脚本目录到路径，以便导入模型定义
sys.path.append(str(Path(__file__).parent))
from hierarchical_model import HierarchicalMultimodalModel

# --- 配置 ---
MODEL_PATH = r"E:\毕设数据集\multimodal_work\models\best_multimodal_model.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_SEQ_LEN = 20

def predict_user(user_dir):
    user_path = Path(user_dir)
    timeline_file = user_path / "timeline_cleaned.jsonl"
    
    if not timeline_file.exists():
        print(f"错误：找不到该用户的 timeline_cleaned.jsonl 文件: {timeline_file}")
        return

    # 1. 准备模型
    print("正在加载模型和分词器...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = HierarchicalMultimodalModel(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. 构建预测序列
    all_tweets = []
    with open(timeline_file, 'r', encoding='utf-8') as f:
        for line in f:
            all_tweets.append(json.loads(line))
    
    # 模拟 20 条推文序列
    tweets = all_tweets[-MAX_SEQ_LEN:]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_ids_list, attn_mask_list, pixels_list, seq_mask = [], [], [], []

    for i in range(MAX_SEQ_LEN):
        if i < len(tweets):
            t = tweets[i]
            enc = tokenizer(t['text'], padding='max_length', truncation=True, max_length=64, return_tensors='pt')
            input_ids_list.append(enc['input_ids'].squeeze(0))
            attn_mask_list.append(enc['attention_mask'].squeeze(0))
            
            if t['has_image']:
                img_path = user_path / f"{t['tweet_id']}.jpg"
                try:
                    img = Image.open(img_path).convert('RGB')
                    pixels_list.append(transform(img))
                except:
                    pixels_list.append(torch.zeros(3, 224, 224))
            else:
                pixels_list.append(torch.zeros(3, 224, 224))
            seq_mask.append(True)
        else:
            input_ids_list.append(torch.zeros(64, dtype=torch.long))
            attn_mask_list.append(torch.zeros(64, dtype=torch.long))
            pixels_list.append(torch.zeros(3, 224, 224))
            seq_mask.append(False)

    # 3. 推理
    batch_ids = torch.stack(input_ids_list).unsqueeze(0).to(DEVICE)
    batch_mask = torch.stack(attn_mask_list).unsqueeze(0).to(DEVICE)
    batch_pixels = torch.stack(pixels_list).unsqueeze(0).to(DEVICE)
    batch_s_mask = torch.tensor(seq_mask).unsqueeze(0).to(DEVICE)

    print("正在进行风险评估...")
    with torch.no_grad():
        logits = model(batch_ids, batch_mask, batch_pixels, batch_s_mask)
        probs = F.softmax(logits, dim=1)
        risk_score = probs[0][1].item()
    
    result = "抑郁 (Positive)" if risk_score > 0.5 else "非抑郁 (Negative)"
    print(f"\n" + "="*30)
    print(f"用户 ID: {user_path.name}")
    print(f"判定结果: {result}")
    print(f"抑郁风险评分: {risk_score*100:.2f}%")
    print(f"参与评估的推文数: {len(tweets)}")
    print("="*30)

if __name__ == "__main__":
    # 默认测试 positive 类别中的一个用户
    TEST_USER = r"E:\毕设数据集\multimodal_work\data\cleaned_data\positive\______Sun______"
    
    if len(sys.argv) > 1:
        predict_user(sys.argv[1])
    else:
        predict_user(TEST_USER)
