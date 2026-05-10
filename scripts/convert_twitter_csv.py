import pandas as pd
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def write_user_data(user_id, group, label_dir):
    user_id_str = str(user_id)
    user_dir = label_dir / user_id_str
    user_dir.mkdir(exist_ok=True)
    
    timeline_path = user_dir / 'timeline_cleaned.jsonl'
    
    # Check current number of lines to maintain tweet_id index
    # Note: In a multithreaded environment within a single chunk, 
    # we don't need to count existing lines if we assume 
    # this function is the only one writing to this user in this chunk.
    # However, across chunks, we need to know the starting index.
    # To keep it simple and fast, we'll append and use a simpler ID if needed, 
    # but the current script uses an index. 
    # Let's pass the current count or just use a timestamp-based ID?
    # The previous script used user_tweet_count.
    
    # To be thread-safe across chunks (if we ever parallelized chunks), 
    # we'd need locks. But since we process chunks sequentially, 
    # we only need to manage the index.
    
    output_lines = []
    for i, (_, row) in enumerate(group.iterrows()):
        tweet_data = {
            "tweet_id": f"{user_id_str}_part", # Simplified ID to avoid global counter complexity in threads
            "user_id": user_id_str,
            "time": None,
            "text": str(row['text']),
            "has_image": False
        }
        output_lines.append(json.dumps(tweet_data, ensure_ascii=False) + '\n')
    
    with open(timeline_path, 'a', encoding='utf-8') as f:
        f.writelines(output_lines)
    
    return len(group)

def convert_csv_to_multimodal(csv_path, output_base, target_label, source_filter_value, max_workers=8):
    print(f"Processing {csv_path} (filtering for '{source_filter_value}') into {target_label} using {max_workers} threads...")
    label_dir = output_base / target_label
    label_dir.mkdir(parents=True, exist_ok=True)
    
    chunksize = 200000
    total_tweets = 0
    
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    # Use ThreadPoolExecutor for I/O bound tasks (writing to many files)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, on_bad_lines='skip', low_memory=False):
            filtered_chunk = chunk[chunk['disorder'] == source_filter_value]
            if filtered_chunk.empty:
                continue

            groups = list(filtered_chunk.groupby('user_id'))
            
            # Submit each user group in the chunk to the thread pool
            futures = [executor.submit(write_user_data, user_id, group, label_dir) for user_id, group in groups]
            
            for future in futures:
                total_tweets += future.result()
            
            print(f"  Processed {total_tweets} tweets...")

def main():
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    output_base = project_root / "data" / "twitter_data"
    twitter_base = Path(r"E:\毕设数据集\twitter")
    
    # We can process the two main files sequentially, but use threads inside each.
    # Alternatively, we could use ProcessPoolExecutor to run these two in parallel.
    # Given the high I/O, sequential chunk reading with internal threading is safer for the disk.
    
    disorder_csv = twitter_base / "anon_disorder_tweets" / "anon_disorder_tweets.csv"
    convert_csv_to_multimodal(disorder_csv, output_base, "positive", "depression")
        
    control_csv = twitter_base / "anon_control_tweets" / "anon_control_tweets.csv"
    convert_csv_to_multimodal(control_csv, output_base, "negative", "control")

    print("\nConversion finished!")

if __name__ == "__main__":
    main()
