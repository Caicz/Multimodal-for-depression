import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# 设置中文字体 (防止可视化乱码)
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 添加脚本目录到路径，以便导入模型定义
sys.path.append(str(Path(__file__).parent))
from hierarchical_model import HierarchicalMultimodalModel

# --- 配置 ---
MODEL_PATH_TWITTER = "models/best_model_twitter.pt"
MODEL_PATH_SWDD = "models/best_model_swdd.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_SEQ_LEN = 25
TEXT_MAX_LEN = 128

SYMPTOM_LABELS = [
    "A_Depressed_Mood", "B_Anhedonia", "C_Weight_Appetite", 
    "D_Sleep_Disturbance", "E_Psychomotor", "F_Fatigue", 
    "G_Guilt_Worthlessness", "H_Concentration", "I_Suicidal_Ideation"
]

DSM5_DIMENSIONS_EN = {
    "A_Depressed_Mood": "I feel so incredibly low, unhappy, and hollow inside. Every day is a struggle against this overwhelming sadness and despair that I just can't shake off; it feels like I've fallen into a bottomless pit and I'm suffocating in the dark.",
    "B_Anhedonia": "I've lost interest in everything I used to love and nothing brings me joy anymore. The world feels completely dull and gray, and I feel like a walking corpse because I just don't care about anything at all.",
    "C_Weight_Appetite": "My appetite is totally gone and I feel nauseous just looking at food, or I find myself uncontrollably binge eating to try and fill the painful emptiness inside me. My weight is changing so fast and I feel disgusted with myself.",
    "D_Sleep_Disturbance": "I can't sleep no matter how tired I am, tossing and turning all night and staring at the ceiling at 4 AM. Or I just want to sleep forever and never wake up because it's the only way to escape this reality.",
    "E_Psychomotor": "My brain feels like it's moving in slow motion and I can't think straight, or I feel so restless and agitated that I want to crawl out of my own skin. I'm constantly on edge and my emotions are always out of control.",
    "F_Fatigue": "I feel completely exhausted and powerless every single day, as if even breathing is a massive effort. I have zero energy left and I just want to disappear because I'm too tired to keep going.",
    "G_Guilt_Worthlessness": "I feel like a total failure and a useless burden to everyone around me. I hate myself and I'm constantly drowning in guilt, believing that everything that goes wrong is my fault and I don't deserve anything good.",
    "H_Concentration": "I can't focus on anything and my mind feels like total mush. I struggle to make even the simplest decisions and I find myself just spacing out for hours because my brain feels like it's completely stopped working.",
    "I_Suicidal_Ideation": "I just want to end it all and finally find some peace away from this pain. I keep thinking about death and how much better off everyone would be without me; I'm ready to say my final farewell and just leave this world."
}

class DimensionRecoverer:
    def __init__(self, lang='en'):
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)
        self.descs = list(DSM5_DIMENSIONS_EN.values())
        self.keys = list(DSM5_DIMENSIONS_EN.keys())
        self.dim_embeddings = self.model.encode(self.descs, convert_to_tensor=True)

    def get_dimension_scores(self, texts):
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        cos_sims = util.cos_sim(embeddings, self.dim_embeddings)
        return cos_sims.cpu().numpy()

class VisualizationManager:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_fig_5_1(self, transformer_attn, user_id):
        """图 5-1：抑郁用户时序注意力热力图"""
        # transformer_attn 是一个列表，包含每一层的注意力 Tensor [Batch, Seq, Seq]
        # 取最后一层的注意力，并取第一个 Batch
        last_layer_attn = transformer_attn[-1][0].cpu().numpy() # [26, 26]
        # 取 CLS 标记 (索引 0) 对 25 条推文 (索引 1:26) 的注意力
        cls_attn = last_layer_attn[0, 1:26]
        
        plt.figure(figsize=(12, 3))
        plt.imshow(cls_attn.reshape(1, -1), cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Attention Weight')
        plt.title(f"用户 {user_id} 时序注意力热力图 (CLS 标记)")
        plt.xlabel("推文序列索引 (时间从远到近)")
        plt.xticks(range(25), range(1, 26))
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{user_id}_fig_5_1.png")
        plt.close()

    def generate_fig_5_2(self, dim_scores, risk_density, user_id):
        """图 5-2：九大 DSM-5 症状维度决策贡献得分柱状图"""
        # dim_scores: [Seq, 9]
        # 贡献得分 = 该维度在所有推文中的累积相似度得分
        contrib = np.sum(dim_scores, axis=0)
        # 归一化并结合风险密度
        if np.sum(contrib) > 0:
            contrib = (contrib / np.sum(contrib)) * risk_density
            
        plt.figure(figsize=(10, 5))
        short_labels = [s.split('_')[1] for s in SYMPTOM_LABELS]
        colors = plt.cm.viridis(np.linspace(0, 0.8, 9))
        plt.bar(short_labels, contrib, color=colors)
        plt.xticks(rotation=45)
        plt.ylabel("决策贡献得分")
        plt.title(f"用户 {user_id} 九大 DSM-5 症状维度决策贡献分析")
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{user_id}_fig_5_2.png")
        plt.close()

    def generate_fig_5_3(self, dim_scores, cross_attn_weights, user_id):
        """图 5-3：不同症状维度平均图像贡献度 对比图"""
        # cross_attn_weights: [B*S, 1, 1] -> [Seq]
        weights = cross_attn_weights.view(-1).cpu().numpy()
        symptom_img_attn = [[] for _ in range(9)]
        
        for i in range(len(weights)):
            if i < dim_scores.shape[0]:
                # 如果推文在该维度上有显著得分 (阈值 0.35)
                for d in range(9):
                    if dim_scores[i, d] > 0.35:
                        symptom_img_attn[d].append(weights[i])
        
        avg_attn = [np.mean(vals) if vals else 0 for vals in symptom_img_attn]
        
        plt.figure(figsize=(10, 5))
        short_labels = [s.split('_')[1] for s in SYMPTOM_LABELS]
        plt.bar(short_labels, avg_attn, color='salmon')
        plt.xticks(rotation=45)
        plt.ylabel("平均图像注意力权重")
        plt.title(f"用户 {user_id} 不同症状维度的平均图像贡献度对比")
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{user_id}_fig_5_3.png")
        plt.close()

    def generate_fig_5_4(self, pos_infos, neg_infos):
        """图 5-4：10+ 抑郁用户与对照组用户的风险分数曲线对比"""
        plt.figure(figsize=(14, 7))
        
        # 绘制负样本 (对照组) - 蓝色细线
        for i, (user_id, seq) in enumerate(neg_infos):
            times, scores = self._get_seq_data(seq)
            if len(times) == 0: continue
            # 将时间归一化为相对天数 (从该序列第一条推文开始)
            start_time = min(times)
            rel_days = [(t - start_time).total_seconds() / 86400.0 for t in times]
            
            label = "对照组 (Negative)" if i == 0 else None
            plt.plot(rel_days, scores, color='blue', alpha=0.15, linewidth=1, label=label)
            plt.scatter(rel_days, scores, color='blue', alpha=0.1, s=10)

        # 绘制正样本 (抑郁组) - 红色细线
        for i, (user_id, seq) in enumerate(pos_infos):
            times, scores = self._get_seq_data(seq)
            if len(times) == 0: continue
            start_time = min(times)
            rel_days = [(t - start_time).total_seconds() / 86400.0 for t in times]
            
            label = "抑郁组 (Positive)" if i == 0 else None
            plt.plot(rel_days, scores, color='red', alpha=0.3, linewidth=1.5, label=label)
            plt.scatter(rel_days, scores, color='red', alpha=0.2, s=15)

        plt.xlabel("相对时间 (天)")
        plt.ylabel("DSM-5 风险分数 (相似度加权)")
        plt.title("抑郁组与对照组用户风险演化轨迹对比 (N=20)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.output_dir / "fig_5_4_population_comparison.png")
        plt.close()

    def _get_seq_data(self, seq):
        times = []
        scores = []
        for t in seq:
            try:
                dt = datetime.strptime(t['time'], "%Y-%m-%d %H:%M:%S")
                times.append(dt)
                scores.append(t.get('risk_score', 0))
            except: continue
        if not times: return [], []
        # 按时间排序
        idx = np.argsort(times)
        return np.array(times)[idx].tolist(), np.array(scores)[idx].tolist()

def predict_user_with_viz(user_data, model, tokenizer, vlm_tokenizer, recoverer, viz, gen_individual=False):
    user_id = user_data['user_id']
    sequence = user_data['sequence'][:MAX_SEQ_LEN]
    risk_density = user_data.get('risk_density', 0.0)
    
    # 1. 准备 Tensor 输入
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_ids_list, attn_mask_list, pixels_list, rt_list, seq_mask = [], [], [], [], []
    project_root = Path(__file__).parent.parent.absolute()
    texts = []

    for i in range(MAX_SEQ_LEN):
        if i < len(sequence):
            t = sequence[i]
            texts.append(t['text'])
            enc = tokenizer(t['text'], padding='max_length', truncation=True, max_length=TEXT_MAX_LEN, return_tensors='pt')
            input_ids_list.append(enc['input_ids'].squeeze(0))
            attn_mask_list.append(enc['attention_mask'].squeeze(0))
            rt_list.append(1 if t.get('is_retweet') else 0)
            
            if t.get('has_image'):
                img_path = project_root / t['user_path'] / f"{t['tweet_id']}.jpg"
                try:
                    img = Image.open(img_path).convert('RGB')
                    pixels_list.append(transform(img))
                except: pixels_list.append(torch.zeros(3, 224, 224))
            else:
                pixels_list.append(torch.zeros(3, 224, 224))
            seq_mask.append(True)
        else:
            texts.append("")
            input_ids_list.append(torch.zeros(TEXT_MAX_LEN, dtype=torch.long))
            attn_mask_list.append(torch.zeros(TEXT_MAX_LEN, dtype=torch.long))
            pixels_list.append(torch.zeros(3, 224, 224))
            rt_list.append(0)
            seq_mask.append(False)

    batch_ids = torch.stack(input_ids_list).unsqueeze(0).to(DEVICE)
    batch_mask = torch.stack(attn_mask_list).unsqueeze(0).to(DEVICE)
    batch_pixels = torch.stack(pixels_list).unsqueeze(0).to(DEVICE)
    batch_rts = torch.tensor(rt_list).unsqueeze(0).to(DEVICE)
    batch_s_mask = torch.tensor(seq_mask).unsqueeze(0).to(DEVICE)
    batch_density = torch.tensor([risk_density]).to(DEVICE)
    
    # VLM 摘要
    vlm_text = user_data.get('vlm_summary', "")
    s_enc = vlm_tokenizer(vlm_text, padding='max_length', truncation=True, max_length=TEXT_MAX_LEN, return_tensors='pt')
    sum_ids = s_enc['input_ids'].to(DEVICE)
    sum_mask = s_enc['attention_mask'].to(DEVICE)

    # 2. 运行推理 (即使不生成图也需要推理来获取注意力/分数)
    with torch.no_grad():
        logits, transformer_attn, cross_attn = model(
            batch_ids, batch_mask, batch_pixels, batch_s_mask, 
            batch_density, batch_rts, 
            summary_ids=sum_ids, summary_mask=sum_mask, 
            return_attn=True
        )
        prob = F.softmax(logits, dim=1)[0][1].item()

    # 3. 如果需要个案图，则进行恢复和绘图
    if gen_individual:
        print(f"生成用户 {user_id} 的个案可解释性图表...")
        dim_scores = recoverer.get_dimension_scores(texts)
        viz.generate_fig_5_1(transformer_attn, user_id)
        viz.generate_fig_5_2(dim_scores, risk_density, user_id)
        viz.generate_fig_5_3(dim_scores, cross_attn, user_id)
    
    return user_id, sequence

def main():
    import random
    # 1. 初始化
    dataset = 'twitter'
    model_path = MODEL_PATH_TWITTER
    bert_id = 'bert-base-uncased'
    
    tokenizer = AutoTokenizer.from_pretrained(bert_id)
    vlm_tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    recoverer = DimensionRecoverer()
    
    model = HierarchicalMultimodalModel(num_classes=2, bert_model_name=bert_id).to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"已加载预训练模型: {model_path}")
    model.eval()

    viz = VisualizationManager("results/visualizations")
    
    # 2. 加载数据
    data_path = f"data/user_sequences_{dataset}_with_vlm_test.jsonl"
    if not os.path.exists(data_path):
        data_path = f"data/user_sequences_{dataset}_with_vlm.jsonl"
        
    print(f"从 {data_path} 加载并随机采样样本...")
    
    all_positive = []
    all_negative = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data['label'] == 1:
                all_positive.append(data)
            else:
                all_negative.append(data)
    
    # 随机采样 10 个正样本和 10 个负样本
    sample_pos = random.sample(all_positive, min(10, len(all_positive)))
    sample_neg = random.sample(all_negative, min(10, len(all_negative)))
    
    print(f"采样完成: 正样本 {len(sample_pos)} 个, 负样本 {len(sample_neg)} 个")

    # 3. 处理样本
    pos_results = []
    for i, data in enumerate(sample_pos):
        # 仅为第一个样本生成个案图
        res = predict_user_with_viz(data, model, tokenizer, vlm_tokenizer, recoverer, viz, gen_individual=(i==0))
        pos_results.append(res)
    
    neg_results = []
    for i, data in enumerate(sample_neg):
        res = predict_user_with_viz(data, model, tokenizer, vlm_tokenizer, recoverer, viz, gen_individual=(i==0))
        neg_results.append(res)
    
    # 4. 生成群体对比图 (Fig 5-4)
    print("\n正在生成 10 vs 10 群体风险演化对比图...")
    viz.generate_fig_5_4(pos_results, neg_results)
    
    print(f"\n可视化全流程完成！结果保存至: {viz.output_dir.absolute()}")

if __name__ == "__main__":
    main()
