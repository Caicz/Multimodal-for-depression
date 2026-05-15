import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def build_sequences(data_root, high_risk_file, output_file, max_len=25, hr_ratio=0.8, ablation=False):
    project_root = Path(__file__).parent.parent.absolute()
    data_root = (project_root / data_root).absolute()
    high_risk_file = (project_root / high_risk_file).absolute()
    output_file = (project_root / output_file).absolute()

    if not data_root.exists():
        print(f"跳过: 找不到数据目录 {data_root}")
        return

    sequences = []
    if not ablation:
        # 1. 建立高风险推文的 ID -> risk_score 的映射
        tid_to_score = {} 
        hr_users = defaultdict(list)
        
        if high_risk_file.exists():
            with open(high_risk_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        tid = str(item['tweet_id'])
                        score = float(item.get('risk_score', 0))
                        tid_to_score[tid] = score
                        hr_users[item['user_id']].append(tid)
                    except: continue
                
        # 2. 扫描所有用户
        all_users = []
        for label in ["positive", "negative"]:
            label_dir = data_root / label
            if not label_dir.exists(): continue
            for user_dir in label_dir.iterdir():
                if user_dir.is_dir() and (user_dir / 'timeline_cleaned.jsonl').exists():
                    all_users.append({
                        "user_id": user_dir.name,
                        "user_path": user_dir.absolute(),
                        "label": 1 if label == "positive" else 0
                    })

        print(f"总用户数: {len(all_users)}, 其中有风险信号的用户: {len(hr_users)}")

        time_format = "%Y-%m-%d %H:%M:%S"

        for user_info in all_users:
            user_id = user_info['user_id']
            abs_user_path = user_info['user_path']
            label = user_info['label']
            user_rel_path = abs_user_path.relative_to(project_root)
            
            timeline_file = abs_user_path / 'timeline_cleaned.jsonl'
            all_tweets = []
            with open(timeline_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        t_data = json.loads(line)
                        # 识别是否为转发 (Retweet)
                        t_data['is_retweet'] = t_data['text'].startswith("RT @")
                        # 关键点：给原始推文打上风险分 (如果有的话)
                        tid = str(t_data['tweet_id'])
                        t_data['risk_score'] = tid_to_score.get(tid, 0.0)
                        all_tweets.append(t_data)
                    except: continue
            
            if not all_tweets: continue

            # --- 采样策略 (Greedy Priority) ---
            # 区分风险推文
            hr_pool = [t for t in all_tweets if t['risk_score'] > 0]
            # 按分数降序排列
            hr_pool.sort(key=lambda x: x['risk_score'], reverse=True)
            
            # 尽可能选满 max_len (20) 条高风险推文
            selected_hr = hr_pool[:max_len]
            hr_ids = {str(t['tweet_id']) for t in selected_hr}
            
            # 补齐：只有当风险推文不足 max_len 时才使用普通推文
            needed = max_len - len(selected_hr)
            if needed > 0:
                # 在剩余池中随机选择普通推文或未被选中的低分风险推文
                remaining_pool = [t for t in all_tweets if str(t['tweet_id']) not in hr_ids]
                if len(remaining_pool) > needed:
                    selected_fill = random.sample(remaining_pool, needed)
                else:
                    selected_fill = remaining_pool
            else:
                selected_fill = []
                
            user_seq = selected_hr + selected_fill
            
            # --- 计算风险密度 ---
            seq_times = []
            total_risk = 0.0
            for t in user_seq:
                t['user_path'] = str(user_rel_path).replace("\\", "/")
                total_risk += t['risk_score']
                try:
                    if(t['time'] == null):
                        continue
                    dt = datetime.strptime(t['time'], time_format)
                    seq_times.append(dt)
                except: continue
            
            if seq_times:
                days = (max(seq_times) - min(seq_times)).total_seconds() / 86400.0
                risk_density = total_risk / max(1.0, days)
            else:
                days = 0
                risk_density = 0.0
            if seq_times:
                user_seq.sort(key=lambda x: x['time'])
            
            sequences.append({
                "user_id": user_id,
                "label": label,
                "risk_density": round(float(risk_density), 6),
                "seq_time_span_days": round(float(days), 2),
                "sequence": user_seq,
                "seq_len": len(user_seq)
            })
    else:
        all_users = []
        for label in ["positive", "negative"]:
            label_dir = data_root / label
            if not label_dir.exists(): continue
            for user_dir in label_dir.iterdir():
                if user_dir.is_dir() and (user_dir / 'timeline_cleaned.jsonl').exists():
                    all_users.append({
                        "user_id": user_dir.name,
                        "user_path": user_dir.absolute(),
                        "label": 1 if label == "positive" else 0
                    })
        
        print(f"总用户数: {len(all_users)}")
        time_format = "%Y-%m-%d %H:%M:%S"

        for user_info in all_users:
            user_id = user_info['user_id']
            abs_user_path = user_info['user_path']
            label = user_info['label']
            user_rel_path = abs_user_path.relative_to(project_root)
            
            timeline_file = abs_user_path / 'timeline_cleaned.jsonl'
            all_tweets = []
            with open(timeline_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        t_data = json.loads(line)
                        # 识别是否为转发 (Retweet)
                        t_data['is_retweet'] = t_data['text'].startswith("RT @")
                        # 关键点：给原始推文打上风险分 (如果有的话)
                        tid = str(t_data['tweet_id'])
                        t_data['risk_score'] = 0.0  # 消融版本不使用风险分
                        all_tweets.append(t_data)
                    except:
                        print(f"警告: 无法解析推文数据，跳过该行。用户ID: {user_id}, 用户路径: {user_rel_path}")
                        continue
            
            if not all_tweets: continue

            # --- 采样策略 (Greedy Priority) ---
            
            remaining_pool = [t for t in all_tweets]
            if len(remaining_pool) > max_len:
                selected_fill = random.sample(remaining_pool, max_len)
            else:
                selected_fill = remaining_pool

                
            user_seq =  selected_fill
            
            # --- 计算风险密度 ---
            seq_times = []
            total_risk = 0.0
            for t in user_seq:
                t['user_path'] = str(user_rel_path).replace("\\", "/")
                total_risk += t['risk_score']
                try:
                    if(t['time'] == null):
                        continue
                    dt = datetime.strptime(t['time'], time_format)
                    seq_times.append(dt)
                except: continue
            
            if seq_times:
                days = (max(seq_times) - min(seq_times)).total_seconds() / 86400.0
                risk_density = total_risk / max(1.0, days)
            else:
                days = 0
                risk_density = 0.0
            if seq_times:
                user_seq.sort(key=lambda x: x['time'])
            
            sequences.append({
                "user_id": user_id,
                "label": label,
                "risk_density": round(float(risk_density), 6),
                "seq_time_span_days": round(float(days), 2),
                "sequence": user_seq,
                "seq_len": len(user_seq)
            })

    with open(output_file, 'w', encoding='utf-8') as f:
        for seq in sequences:
            f.write(json.dumps(seq, ensure_ascii=False) + '\n')
            
    print(f"序列数据集生成完毕: {output_file.name}, 总计 {len(sequences)} 个用户。")

def main():
    parser = argparse.ArgumentParser(description="构建用户推文序列数据集")
    parser.add_argument('--dataset', type=str, default='swdd', choices=['swdd', 'twitter', 'all'],
                        help='指定处理的数据集: swdd, twitter 或 all')
    parser.add_argument('--ablation', action='store_true', help='是否生成消融版本(随机采样而非优先采样)')
    args = parser.parse_args()

    data_dir = Path("data")
    tasks = []
    if args.ablation:
        if args.dataset in ['swdd', 'all']:
            tasks.append({"root": "data/swdd_data", "in": "data/high_risk_tweets_swdd.jsonl", "out": "data/user_sequences_swdd_ablation.jsonl"})
        if args.dataset in ['twitter', 'all']:
            tasks.append({"root": "data/twitter_data", "in": "data/high_risk_tweets_twitter.jsonl", "out": "data/user_sequences_twitter_ablation.jsonl"})
    else:
        if args.dataset in ['swdd', 'all']:
            tasks.append({"root": "data/swdd_data", "in": "data/high_risk_tweets_swdd.jsonl", "out": "data/user_sequences_swdd.jsonl"})
        if args.dataset in ['twitter', 'all']:
            tasks.append({"root": "data/twitter_data", "in": "data/high_risk_tweets_twitter.jsonl", "out": "data/user_sequences_twitter.jsonl"})

    for task in tasks:
        build_sequences(task['root'], task['in'], task['out'], ablation=args.ablation)

if __name__ == "__main__":
    main()
