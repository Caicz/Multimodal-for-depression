import json
import re
import html
import os
from pathlib import Path
from tqdm import tqdm

def clean_tweet_text(text):
    if not text:
        return ""
    # 1. 剔除 HTTP/HTTPS 链接
    text = re.sub(r'https?://\S+', '', text)
    # 2. 修复 HTML 转义字符 (如 &amp; -> &)
    text = html.unescape(text)
    # 3. 剔除多余的换行符和空格
    text = ' '.join(text.split())
    return text

def process_directory(base_dir):
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"跳过：目录 {base_dir} 不存在")
        return

    # 查找所有需要清洗的原始文件
    files = list(base_path.glob("**/timeline_cleaned.jsonl"))
    print(f"在 {base_dir} 中找到 {len(files)} 个待处理文件...")

    for f_path in tqdm(files, desc="清洗进度"):
        temp_path = f_path.with_suffix(".tmp")
        try:
            with open(f_path, 'r', encoding='utf-8', errors='ignore') as f_in, \
                 open(temp_path, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    try:
                        data = json.loads(line)
                        data['text'] = clean_tweet_text(data.get('text', ''))
                        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    except:
                        continue
            
            # 安全替换原文件
            os.replace(temp_path, f_path)
        except Exception as e:
            print(f"处理文件 {f_path} 时出错: {e}")
            if temp_path.exists():
                temp_path.unlink()

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.absolute()
    
    # 分别处理 Twitter 和 SWDD 数据集
    process_directory(project_root / "data" / "twitter_data")
    
    print("\n所有原始推文噪声处理完成！")
