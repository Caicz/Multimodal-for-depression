import time
import json
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 测试配置 ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3.5:9b"  # 确保此模型已 pull
TEST_TEXT = "我年华虚度，空有一身疲倦。"
TEST_PROMPT = f"Summary(50w): {TEST_TEXT}"

def single_request(session):
    """发送单个推理请求并返回耗时"""
    payload = {
        "model": MODEL_NAME,
        "prompt": TEST_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 50  # 限制输出长度以保持测试一致性
        }
    }
    start = time.time()
    try:
        response = session.post(OLLAMA_API_URL, json=payload, timeout=60)
        if response.status_code == 200:
            end = time.time()
            data = response.json()
            # 获取模型实际生成的 token 数量
            tokens = data.get("eval_count", 0)
            return end - start, tokens
    except Exception as e:
        return None, 0
    return None, 0

def benchmark(workers, total_requests):
    print(f"=== 开始性能测试 ===")
    print(f"模型: {MODEL_NAME}")
    print(f"并发数 (Workers): {workers}")
    print(f"总请求数: {total_requests}")
    print(f"提示词: {TEST_PROMPT}")
    print("-" * 30)

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
    session.mount("http://", adapter)

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(single_request, session) for _ in range(total_requests)]
        
        completed = 0
        for future in as_completed(futures):
            duration, tokens = future.result()
            if duration:
                results.append((duration, tokens))
            completed += 1
            if completed % (total_requests // 10 or 1) == 0:
                print(f"已完成: {completed}/{total_requests}...")

    end_time = time.time()
    total_duration = end_time - start_time
    
    # 统计数据
    valid_results = [r for r in results if r[0] is not None]
    if not valid_results:
        print("所有请求均失败，请检查 Ollama 服务和模型名称。")
        return

    avg_latency = sum(r[0] for r in valid_results) / len(valid_results)
    total_tokens = sum(r[1] for r in valid_results)
    
    print("-" * 30)
    print(f"测试完成!")
    print(f"总耗时: {total_duration:.2f} 秒")
    print(f"平均延迟: {avg_latency:.2f} 秒/请求")
    print(f"吞吐量 (每秒处理推文数): {len(valid_results) / total_duration:.2f} tweets/s")
    print(f"生成速度 (每秒生成 Token 数): {total_tokens / total_duration:.2f} tokens/s")
    print(f"成功率: {len(valid_results)}/{total_requests}")
    print("-" * 30)
    print("提示: 如果 GPU 利用率低，请尝试增加 --workers 数值。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8, help='并发请求数')
    parser.add_argument('--total', type=int, default=100, help='总测试请求数')
    parser.add_argument('--model', type=str, default="qwen3.5:9b")
    args = parser.parse_args()
    
    MODEL_NAME = args.model
    benchmark(args.workers, args.total)
