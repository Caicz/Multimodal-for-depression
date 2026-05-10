import json
import os
from pathlib import Path

def count_labels(file_path):
    pos = 0
    neg = 0
    total = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                label = data.get('label')
                if label == 1:
                    pos += 1
                elif label == 0:
                    neg += 1
                total += 1
        return pos, neg, total
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    data_dir = Path('data')
    files = list(data_dir.glob('user_sequences_*.jsonl'))
    
    print(f"{'File Name':<50} | {'Positive':<10} | {'Negative':<10} | {'Total':<10}")
    print("-" * 90)
    
    for file_path in sorted(files):
        res = count_labels(file_path)
        if res:
            pos, neg, total = res
            print(f"{file_path.name:<50} | {pos:<10} | {neg:<10} | {total:<10}")

if __name__ == "__main__":
    main()
