import json
import os
import base64
import requests
import argparse
import hashlib
import time
import sqlite3
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- 配置 ---
OLLAMA_API_URL = "http://localhost:11435/api/generate"
MODEL_NAME = "qwen3.5:9b" 
MAX_RETRIES = 2
CACHE_DB = "data/vlm_cache.db"

# 初始化持久化缓存 (SQLite 是标准库，无需安装)
def init_cache():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cache (
            hash TEXT PRIMARY KEY,
            summary TEXT
        )
    ''')
    conn.commit()
    conn.close()

def get_cached_summary(content_hash):
    try:
        conn = sqlite3.connect(CACHE_DB)
        cursor = conn.cursor()
        cursor.execute('SELECT summary FROM cache WHERE hash = ?', (content_hash,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except:
        return None

def set_cached_summary(content_hash, summary):
    # 多进程写入时增加重试逻辑
    for _ in range(5):
        try:
            conn = sqlite3.connect(CACHE_DB, timeout=20)
            cursor = conn.cursor()
            cursor.execute('INSERT OR REPLACE INTO cache (hash, summary) VALUES (?, ?)', (content_hash, summary))
            conn.commit()
            conn.close()
            break
        except sqlite3.OperationalError:
            time.sleep(0.5)

def get_content_hash(text, images_base64=None):
    hasher = hashlib.md5(text.encode('utf-8'))
    if images_base64:
        for img in images_base64:
            hasher.update(img.encode('utf-8'))
    return hasher.hexdigest()

def encode_image_raw(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except:
        return None

def get_vlm_summary(tweet_texts, images_base64=None):
    """调用 Ollama API 对推文序列进行整体总结"""
    combined_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(tweet_texts)])
    content_hash = get_content_hash(combined_text, images_base64)
    cached = get_cached_summary(content_hash)
    if cached:
        return cached

    prompt = (
        "你是一位心理健康专家。请分析以下用户按时间顺序发布的一系列推文内容（以及配图，如果有）。\n"
        "请简要总结该用户的整体情绪状态、心理状态的变化趋势以及潜在的心理风险（100字以内）。\n\n"
        f"推文列表：\n{combined_text}"
    )
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": 100,
        "options": {"temperature": 0.1, "num_predict": 200}
    }
    if images_base64:
        payload["images"] = images_base64

    last_error = ""
    for attempt in range(MAX_RETRIES):
        try:
            with requests.Session() as s:
                response = s.post(OLLAMA_API_URL, json=payload, timeout=180)
                if response.status_code == 200:
                    resp_json = response.json()
                    result = resp_json.get("response", "").strip()
                    if result:
                        set_cached_summary(content_hash, result)
                        return result
                    else:
                        last_error = f"API 返回内容为空。完整响应: {resp_json}"
                else:
                    last_error = f"API 错误状态码: {response.status_code}, 响应: {response.text}"
            time.sleep(1)
        except requests.exceptions.Timeout:
            last_error = "请求超时 (Timeout) - 序列总结可能较慢"
            time.sleep(2)
        except Exception as e:
            last_error = f"网络或系统异常: {str(e)}"
            time.sleep(2)
            
    print(f"\n[错误详情] 序列摘要失败 | 原因: {last_error}")
    return "总结失败"

def process_single_user(user_seq_str, project_root_str):
    user_seq = json.loads(user_seq_str)
    project_root = Path(project_root_str)
    
    tweet_texts = []
    images_base64 = []
    
    for tweet in user_seq['sequence']:
        text = tweet.get('text', '')
        tweet_texts.append(text)
        
        if tweet.get('has_image', False):
            rel_path = tweet['user_path']
            img_path = project_root / rel_path / f"{tweet['tweet_id']}.jpg"
            if img_path.exists():
                img_b64 = encode_image_raw(img_path)
                if img_b64:
                    images_base64.append(img_b64)
    
    summary = get_vlm_summary(tweet_texts, images_base64)
    user_seq['vlm_summary'] = summary
    
    return json.dumps(user_seq, ensure_ascii=False)

def process_dataset(input_file, output_file, max_workers, debug=False):
    if not os.path.exists(input_file):
        print(f"跳过: 找不到文件 {input_file}")
        return

    init_cache()
    
    processed_users = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    user_data = json.loads(line)
                    processed_users.add(user_data['user_id'])
                except:
                    continue
    
    all_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data['user_id'] not in processed_users:
                all_data.append(json.dumps(data))
                if debug and len(all_data) >= 100:
                    break

    if not all_data:
        print(f"文件 {input_file.name} 已全部处理完成。")
        return

    total_to_process = len(all_data)
    print(f"\n[{'DEBUG' if debug else 'RUN'}] 处理 {input_file.name}")
    print(f"待处理用户: {total_to_process}, 并发进程: {max_workers}")
    
    project_root = str(Path(__file__).parent.parent.absolute())
    count = 0
    start_time = time.time()

    with open(output_file, 'a', encoding='utf-8') as f_out:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_user, user_str, project_root) for user_str in all_data]
            
            for future in as_completed(futures):
                try:
                    updated_user_json = future.result()
                    f_out.write(updated_user_json + "\n")
                    count += 1
                    
                    if count % 10 == 0 or debug:
                        elapsed = time.time() - start_time
                        speed = count / elapsed
                        remaining = (total_to_process - count) / speed if speed > 0 else 0
                        print(f"进度: {count}/{total_to_process} ({(count/total_to_process)*100:.2f}%) | "
                              f"速度: {speed:.2f} 用户/秒 | 预计剩余: {remaining/60:.1f} 分钟")
                        f_out.flush()
                except Exception as e:
                    print(f"异常: {e}")

def main():
    parser = argparse.ArgumentParser(description="500万级别数据优化预处理脚本 (无三方库依赖版)")
    parser.add_argument('--dataset', type=str, default='all', choices=['swdd', 'twitter', 'all'])
    parser.add_argument('--workers', type=int, default=64, help='并行进程数')
    parser.add_argument('--debug', action='store_true', help='仅处理 100 条')
    args = parser.parse_args()

    data_dir = Path("data")
    tasks = []
    if args.dataset in ['swdd', 'all']:
        tasks.append({"in": data_dir / "user_sequences_swdd.jsonl", "out": data_dir / "user_sequences_swdd_with_vlm.jsonl"})
    if args.dataset in ['twitter', 'all']:
        tasks.append({"in": data_dir / "user_sequences_twitter.jsonl", "out": data_dir / "user_sequences_twitter_with_vlm.jsonl"})

    for task in tasks:
        process_dataset(task['in'], task['out'], args.workers, args.debug)

if __name__ == "__main__":
    main()
