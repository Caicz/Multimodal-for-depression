import json
import os
import argparse
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import multiprocessing as mp

# --- DSM-5 Depression Dimensions (EN & ZH) ---
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

DSM5_DIMENSIONS_ZH = {
    "A_Depressed_Mood": "我感到内心极度空虚、悲伤和抑郁，整个人跌到了谷底，被一种无法摆脱的绝望感所淹没。每一天对我来说都是一种折磨，难过得喘不过气来，感觉自己快要崩溃了。",
    "B_Anhedonia": "我对一切都失去了兴趣，以前喜欢的事现在一点也不想碰。世界在我眼里是灰暗的，我感觉自己就像个活死人，对生活毫无热情，再也体会不到任何快乐了。",
    "C_Weight_Appetite": "我一点食欲都没有，看到食物就反胃；或者我一直在不停地暴饮暴食，试图填补内心的那种空洞感。我的体重变化很大，我讨厌现在的自己。",
    "D_Sleep_Disturbance": "我整晚整晚地失眠，翻来覆去睡不着，看着凌晨四点的世界发呆。或者我只想一直睡下去，永远不要醒来，因为只有在梦里我才能逃避这个现实。",
    "E_Psychomotor": "我的大脑像生锈了一样转不动，反应极度迟钝；或者我感到极度的焦虑和躁动，坐立难安，情绪完全失控，一点点小事就能让我崩溃。",
    "F_Fatigue": "我每天都感到精疲力竭，浑身无力，连呼吸都觉得是一种沉重的负担。我彻底失去了所有的精力，累到想彻底消失，再也没有力气支撑下去了。",
    "G_Guilt_Worthlessness": "我觉得自己是个彻底的失败者，是所有人的累赘。我恨我自己，内心充满了罪恶感，觉得所有发生的不好的事情都是我的错，我不配得到任何幸福。",
    "H_Concentration": "我无法集中精神做任何事，脑子里乱成一团浆糊。我连最简单的决定都犹豫不决，经常对着空气发呆，感觉我的思维已经停滞了。",
    "I_Suicidal_Ideation": "我只想离开这个世界，彻底解脱，再也不想承受这种痛苦了。我脑子里全是关于死亡的想法，觉得没有我大家会过得更好，我已经在计划最后的告别了。"
}

class EmbeddingConfig:
    def __init__(self):
        self.model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        # Force CPU for multi-processing workers to avoid CUDA initialization issues in children
        self.device = "cpu" 
        self.batch_size = 256 # Reduced batch size slightly for memory efficiency in multi-core
        self.similarity_threshold = 0.39 

def filter_tweets_multiprocess(input_dir, output_file, lang='en', num_workers=None):
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    input_dir = Path(input_dir)
    all_tasks = []
    for label in ["positive", "negative"]:
        d = input_dir / label
        if d.exists():
            for u in d.iterdir():
                if u.is_dir() and (u / "timeline_cleaned.jsonl").exists():
                    all_tasks.append((str(u / "timeline_cleaned.jsonl"), str(u), 1 if label == "positive" else 0))

    if not all_tasks:
        print(f"No tasks found in {input_dir}")
        return

    print(f"Starting multi-core filtering for [{input_dir.name}] with {num_workers} workers...")
    
    project_root = Path(__file__).parent.parent.absolute()
    
    # Split tasks into chunks for workers
    chunk_size = max(1, len(all_tasks) // num_workers)
    task_chunks = [all_tasks[i:i + chunk_size] for i in range(0, len(all_tasks), chunk_size)]

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_workers) as pool:
        # Each worker process will filter a subset of users
        worker_args = [(chunk, output_file, lang, project_root) for chunk in task_chunks]
        
        # We use imap_unordered but since each worker writes to its own temp file or we collect, 
        # let's have them write to unique temp files then merge.
        temp_files = pool.starmap(worker_routine, worker_args)

    # Merge temp files into final output_file
    print("Merging temporary results...")
    with open(output_file, "w", encoding="utf-8") as f_out:
        for tmp in temp_files:
            if os.path.exists(tmp):
                with open(tmp, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        f_out.write(line)
                os.remove(tmp)

    with open(output_file, "r", encoding="utf-8") as f:
        final_count = sum(1 for _ in f)
    print(f"Finished! Total high-risk tweets filtered: {final_count}")

def worker_routine(tasks, output_file, lang, project_root):
    # Each process loads its own model and pre-computes dimensions
    cfg = EmbeddingConfig()
    model = SentenceTransformer(cfg.model_name, device=cfg.device)
    
    if lang == 'en':
        dim_descs = list(DSM5_DIMENSIONS_EN.values())
        dim_keys = list(DSM5_DIMENSIONS_EN.keys())
    else:
        dim_descs = list(DSM5_DIMENSIONS_ZH.values())
        dim_keys = list(DSM5_DIMENSIONS_ZH.keys())

    dim_embeddings = model.encode(dim_descs, convert_to_tensor=True)
    
    pid = os.getpid()
    temp_output = f"{output_file}.tmp_{pid}"
    
    count = 0
    with open(temp_output, "w", encoding="utf-8") as f_out:
        batch = []
        for f_path, u_path, lbl in tasks:
            with open(f_path, "r", encoding="utf-8", errors='ignore') as f:
                for line in f:
                    try:
                        tweet = json.loads(line)
                        batch.append((tweet, u_path, lbl))
                        if len(batch) >= cfg.batch_size:
                            count += process_and_write_batch(batch, model, dim_embeddings, dim_keys, cfg, project_root, f_out)
                            batch = []
                    except: continue
        
        if batch:
            count += process_and_write_batch(batch, model, dim_embeddings, dim_keys, cfg, project_root, f_out)
            
    return temp_output

def process_and_write_batch(batch, model, dim_embeddings, dim_keys, cfg, project_root, f_out):
    texts = [t[0].get("text", "") for t in batch]
    if not texts: return 0
    
    # SentenceTransformer.encode is efficient even on CPU for batches
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    cos_sims = util.cos_sim(embeddings, dim_embeddings)
    
    hit_count = 0
    for i in range(len(batch)):
        t_obj, upath, label_val = batch[i]
        scores = cos_sims[i]
        hit_dims = [dim_keys[j] for j, score in enumerate(scores) if score >= cfg.similarity_threshold]
        
        if hit_dims:
            t_obj["user_path"] = str(Path(upath).relative_to(project_root)).replace("\\", "/")
            t_obj["symptoms"] = hit_dims
            t_obj["risk_score"] = len(hit_dims)
            t_obj["label"] = label_val
            t_obj["max_similarity"] = float(torch.max(scores))
            f_out.write(json.dumps(t_obj, ensure_ascii=False) + "\n")
            hit_count += 1
    return hit_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='twitter', choices=['swdd', 'twitter', 'all'])
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes')
    args = parser.parse_args()

    data_base = Path(__file__).parent.parent / "data"
    
    # Optional: ensure multiprocessing starts correctly on Windows
    mp.freeze_support()

    if args.dataset in ['twitter', 'all']:
        filter_tweets_multiprocess(data_base / "twitter_data", data_base / "high_risk_tweets_twitter.jsonl", lang='en', num_workers=args.workers)
    if args.dataset in ['swdd', 'all']:
        filter_tweets_multiprocess(data_base / "swdd_data", data_base / "high_risk_tweets_swdd.jsonl", lang='zh', num_workers=args.workers)

if __name__ == "__main__":
    main()
