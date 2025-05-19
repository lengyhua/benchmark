import os
import random
import requests
import threading
import time
import queue
import base64
import json  # 添加导入json模块
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# 配置
TOTAL_REQUESTS = 80      # 总请求数
CONCURRENCY = [2, 4, 8, 16, 20, 24, 28, 32, 36, 40, 48, 64]  # 并发线程数数组
API_URL = "http://192.168.8.41:8000/v1/chat/completions"  # vLLM OpenAI API 地址
REQUEST_TIMEOUT =3600     # 单次请求超时（秒）
IMAGE_PATH = "pics"  # 测试图片路径
PROMPT = "详细描述这张图片的内容，要求输出在500个字以上，尽量详细！"  # 分析图片的提示词
MODEL_NAME = "Qwen2.5-VL-7B-Instruct"  # 模型名称

# 全局统计
success_count = 0
failed_count = 0
# 如果输出的tokens数过少则认为是不正常的情况
abnormal_count = 0
total_latency = 0
total_tokens = 0
glock = threading.Lock()

request_queue = queue.Queue()

def get_image_paths_from_directory(directory: str) -> list[str]:
    """从目录中获取所有图片文件的路径"""
    supported_extensions = (".jpg", ".jpeg", ".png", ".bmp")  # 支持的图片格式
    image_paths = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.lower().endswith(supported_extensions)
    ]
    if not image_paths:
        raise ValueError(f"目录 {directory} 中没有找到支持的图片文件")
    return image_paths

def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为 Base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_payload(image_base64: str, stream: bool = False) -> dict:
    """构造多模态请求负载"""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 512,
        "stream": stream
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload

def worker(worker_id: int, image_base64: str, stream: bool = True):
    """工作线程：发送图片分析请求"""
    global success_count, failed_count, total_latency, total_tokens, abnormal_count, first_token_latency
    while not request_queue.empty():
        try:
            req_id = request_queue.get_nowait()
        except queue.Empty:
            break

        start_time = time.time()
        try:
            with requests.post(
                API_URL,
                json=build_payload(image_base64, stream),
                timeout=REQUEST_TIMEOUT,
                stream=stream  # 启用流式响应
            ) as response:
                if response.status_code == 200:
                    if stream:
                        first_chunk_time = None
                        content_chunks = []
                        completion_tokens = 0
                        for chunk in response.iter_content(chunk_size=None):
                            if not first_chunk_time:
                                first_chunk_time = time.time()  # 记录首token时间
                            chunk = chunk.decode('utf-8').strip("data: ")
                            try:
                                print(f"Thread-{worker_id}-{req_id} | {chunk}")
                                chunk_json = json.loads(chunk)
                                chunk_choices = chunk_json.get("choices", [])
                                if chunk_choices:
                                    chunk_content = chunk_choices[0]["delta"].get("content", "")
                                    content_chunks.append(chunk_content)  # 收集所有输出
                                else:
                                    completion_tokens = chunk_json.get("usage", {}).get("completion_tokens", 0)
                            except Exception as e:
                                print(f"[ERROR]Thread-{worker_id}-{req_id} | {str(e)}")

                        latency = time.time() - start_time
                        first_token_time = first_chunk_time - start_time if first_chunk_time else latency

                        # 将所有块拼接为完整响应
                        content = "".join(content_chunks)
                        tokens = completion_tokens if completion_tokens>0 else len(content_chunks)
                        
                        with glock:
                            success_count += 1
                            total_latency += latency
                            total_tokens += tokens
                            first_token_latency.append(first_token_time)
                            if tokens <= 10:
                                abnormal_count += 1
                        print(f"[成功]Thread-{worker_id}-{req_id} | "
                            f"耗时: {latency:.2f}s | 首token延迟: {first_token_time:.2f}s | "
                            f"Tokens: {tokens} | Response: {content}")
                    else:
                        content = response.json()
                        latency = time.time() - start_time
                        tokens = content.get("usage", {}).get("completion_tokens", 0)
                        with glock:
                            success_count += 1
                            total_latency += latency
                            total_tokens += tokens
                            if tokens <= 10:
                                abnormal_count += 1
                        print(f"[成功]Thread-{worker_id}-{req_id} | "
                              f"耗时: {latency:.2f}s | Tokens: {tokens} | Response: {content}")
                else:
                    with glock:
                        failed_count += 1
                    print(f"[失败]Thread-{worker_id}-{req_id} | "
                          f"状态码: {response.status_code} | 响应: {response.text}")
        except Exception as e:
            with glock:
                failed_count += 1
            print(f"[异常]Thread-{worker_id}-{req_id} | 错误: {str(e)}")

def run_test(stream: bool = True):
    """启动并发测试"""
    
    # 获取目录中的所有图片路径
    image_paths = get_image_paths_from_directory(IMAGE_PATH)
    print(f"已加载 {len(image_paths)} 张图片用于测试\n")
    
    with open("test_results.md", "w+") as f:
        f.write("# 测试结果\n\n")
        f.write("| 并发数 | 请求数 | 存疑请求数 | 总token输出 | 每图片平均耗时(s) | 首token耗时(s) | token处理速度 | QPS |\n")
        f.write("|--------|--------|--------|-------------|------------------|---------------|----------------|-----|\n")
    
    for concurrency in CONCURRENCY:
        # 初始化任务队列
        for i in range(1, TOTAL_REQUESTS + 1):
            request_queue.put(i)

        print(f"=== 测试并发数: {concurrency} ===")
        print(f"总请求数: {TOTAL_REQUESTS} | 并发数: {concurrency}\n")

        global success_count, failed_count, total_latency, total_tokens, abnormal_count, first_token_latency
        success_count = 0
        failed_count = 0
        total_latency = 0
        total_tokens = 0
        abnormal_count = 0
        first_token_latency = []

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for i in range(concurrency):
                # 随机选择一张图片进行测试
                random_image_path = random.choice(image_paths)
                image_base64 = encode_image_to_base64(random_image_path)
                executor.submit(worker, i + 1, image_base64, stream)
        
        # 统计结果
        with glock:
            total_time = time.time() - start_time
            qps = success_count / total_time if total_time > 0 else 0
            avg_latency = total_latency / success_count if success_count > 0 else 0
            avg_tokens = total_tokens / success_count if success_count > 0 else 0
            avg_first_token_latency = sum(first_token_latency) / len(first_token_latency) if first_token_latency else 0
            vtok = total_tokens / total_time

        print("\n=== 测试结果 ===")
        print(f"成功请求: {success_count} | 失败请求: {failed_count} | 存疑请求: {abnormal_count}")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"平均耗时: {avg_latency:.2f} 秒")
        print(f"平均首token延迟: {avg_first_token_latency:.2f} 秒")
        print(f"总tokens: {total_tokens:.2f}")
        print(f"平均生成 Tokens: {avg_tokens:.1f}")
        print(f"token 速度: {vtok:.2f}\n")
        print(f"QPS: {qps:.5f} (请求/秒)")
        
        with open("test_results.md", "a") as f:
            f.write(f"| {concurrency} | {TOTAL_REQUESTS} | {abnormal_count} | {total_tokens:.2f} | "
                    f"{avg_latency:.2f} | {avg_first_token_latency:.2f} | "
                    f"{vtok:.2f} | {qps:.5f} |\n")
            f.flush()

if __name__ == "__main__":
    run_test(stream=True)
