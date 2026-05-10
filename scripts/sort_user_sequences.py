import json
import argparse
import os
import glob

def sort_jsonl(file_path):
    """
    Reads a JSONL file, sorts the 'sequence' list in each line by 'time',
    and overwrites the original file with the sorted content.
    """
    print(f"Processing: {file_path}...")
    temp_path = file_path + '.tmp'
    count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f_in, \
             open(temp_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if 'sequence' in data and isinstance(data['sequence'], list):
                        # Sort by 'time' field. Lexicographical sort works for YYYY-MM-DD HH:MM:SS
                        data['sequence'].sort(key=lambda x: x.get('time', ''))
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    count += 1
                except json.JSONDecodeError:
                    print(f"Error decoding JSON on a line in {file_path}")
        
        # Replace original with sorted
        os.replace(temp_path, file_path)
        print(f"Successfully sorted {count} users in: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sort tweets in user sequences by time.')
    parser.add_argument('pattern', nargs='?', default='data/user_sequences*.jsonl', 
                        help='File pattern or path (default: data/user_sequences*.jsonl)')
    
    args = parser.parse_args()
    
    # Handle the case where the user might provide a relative path from the root
    # or the scripts directory. We assume it's relative to the project root.
    files = glob.glob(args.pattern)
    
    if not files:
        # Try relative to the script's parent if not found in CWD
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        pattern_in_root = os.path.join(project_root, args.pattern)
        files = glob.glob(pattern_in_root)

    if not files:
        print(f"No files found matching: {args.pattern}")
    else:
        for f in files:
            sort_jsonl(f)
