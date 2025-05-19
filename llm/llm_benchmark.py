import requests
import threading
import time
import queue
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import os

# --- Configuration ---
# These can be overridden by environment variables (see main block)
API_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "your-llm-model-name"
TOTAL_REQUESTS_PER_SETTING = 50
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]
INPUT_TOKEN_LENGTHS = [10, 32, 64, 128, 256, 512]
FIXED_OUTPUT_TOKENS = 512
REQUEST_TIMEOUT = 300  # Seconds

# --- Global Statistics (reset for each test setting) ---
first_token_latencies_list = []
generation_durations_list = []
actual_output_tokens_list = []
metrics_successful_requests = 0
failed_requests_count = 0
stat_lock = threading.Lock()
request_log_file_handle = None # For detailed logging

# --- Helper Functions ---
def generate_prompt(num_tokens: int) -> str:
    """
    Generates a simple prompt with a target number of tokens.
    Note: This is a very rough approximation. Actual token count depends on the model's tokenizer.
    For precise control, use the model's specific tokenizer to prepare prompts.
    """
    return ("text " * num_tokens).strip()

def build_payload(prompt: str, stream: bool = True) -> dict:
    """Constructs the API request payload for chat completions."""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": FIXED_OUTPUT_TOKENS,
        "stream": stream
    }
    if stream:
        # For OpenAI-like APIs, this might enable receiving 'usage' in the stream.
        # Adjust if your API uses a different mechanism.
        payload["stream_options"] = {"include_usage": True}
    return payload

# --- Logging ---
def log_message(message: str, error: bool = False, warning: bool = False, console_only: bool = False):
    """Logs a message to console and optionally to a file."""
    global request_log_file_handle
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_entry = f"[{timestamp}] {message}"
    
    if error:
        print(f"\033[91m{log_entry}\033[0m") # Red for errors
    elif warning:
        print(f"\033[93m{log_entry}\033[0m") # Yellow for warnings
    else:
        print(log_entry)

    if request_log_file_handle and not console_only:
        try:
            request_log_file_handle.write(log_entry + "\n")
            request_log_file_handle.flush()
        except Exception as e:
            print(f"Failed to write to log file: {e}")


# --- Worker Thread ---
def worker(worker_id: int, prompt: str):
    """Sends a single request to the LLM and collects metrics."""
    global metrics_successful_requests, failed_requests_count
    global first_token_latencies_list, generation_durations_list, actual_output_tokens_list

    req_start_time = time.time()
    response_content_parts = []
    
    current_req_first_token_latency = -1.0
    current_req_generation_duration = -1.0
    current_req_output_tokens = 0
    
    first_chunk_arrival_time = None
    last_chunk_arrival_time = None # Time when generation is considered complete for this request

    try:
        log_message(f"Worker {worker_id}: Sending request.", console_only=True)
        with requests.post(
            API_URL,
            json=build_payload(prompt, stream=True),
            timeout=REQUEST_TIMEOUT,
            stream=True
        ) as response:
            if response.status_code == 200:
                for line in response.iter_lines(decode_unicode=True):
                    if line: # Filter out keep-alive new lines or empty lines
                        chunk_arrival_time = time.time()
                        if not first_chunk_arrival_time:
                            first_chunk_arrival_time = chunk_arrival_time
                            current_req_first_token_latency = first_chunk_arrival_time - req_start_time

                        if line.startswith("data: "):
                            data_str = line[len("data: "):].strip()
                            if data_str == "[DONE]":
                                last_chunk_arrival_time = chunk_arrival_time
                                break
                            try:
                                chunk_json = json.loads(data_str)
                                choices = chunk_json.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        response_content_parts.append(content)
                                
                                # Check for usage information in the stream
                                usage_data = chunk_json.get("usage")
                                if usage_data and usage_data.get("completion_tokens") is not None:
                                    current_req_output_tokens = usage_data["completion_tokens"]
                                    # Consider this the end of generation if usage is present
                                    last_chunk_arrival_time = chunk_arrival_time 
                                    # If finish_reason is also present, it might be a more definitive end
                                    if choices[0].get("finish_reason") is not None:
                                        break 

                            except json.JSONDecodeError:
                                log_message(f"Worker {worker_id}: JSONDecodeError for chunk: {data_str}", error=True)
                        else:
                             log_message(f"Worker {worker_id}: Received non-SSE line: {line}", warning=True)
                
                # If stream ended without [DONE] or explicit usage, use the last chunk arrival time
                if first_chunk_arrival_time and not last_chunk_arrival_time:
                    last_chunk_arrival_time = time.time() 

                if first_chunk_arrival_time and last_chunk_arrival_time:
                    current_req_generation_duration = last_chunk_arrival_time - first_chunk_arrival_time
                
                # Fallback for output tokens if not found in stream usage
                if current_req_output_tokens == 0 and response_content_parts:
                    full_response_text = "".join(response_content_parts)
                    # Crude token estimation; replace with actual tokenizer if possible
                    current_req_output_tokens = len(full_response_text.split()) 
                    log_message(f"Worker {worker_id}: Estimated output tokens: {current_req_output_tokens} (usage not found in stream)", warning=True)

                # Validate metrics before recording
                if current_req_first_token_latency > 0 and \
                   current_req_generation_duration >= 0 and \
                   current_req_output_tokens > 0:
                    with stat_lock:
                        metrics_successful_requests += 1
                        first_token_latencies_list.append(current_req_first_token_latency)
                        generation_durations_list.append(current_req_generation_duration)
                        actual_output_tokens_list.append(current_req_output_tokens)
                    log_message(f"Worker {worker_id}: METRICS_OK | FTL: {current_req_first_token_latency:.4f}s | GD: {current_req_generation_duration:.4f}s | OutTokens: {current_req_output_tokens}")
                else:
                    with stat_lock:
                        failed_requests_count += 1
                    log_message(f"Worker {worker_id}: METRICS_INVALID | FTL: {current_req_first_token_latency:.4f}s | GD: {current_req_generation_duration:.4f}s | OutTokens: {current_req_output_tokens} | HTTP: 200", warning=True)
            
            else: # HTTP status not 200
                with stat_lock:
                    failed_requests_count += 1
                log_message(f"Worker {worker_id}: FAILED_HTTP | Status: {response.status_code} | Response: {response.text}", error=True)

    except requests.exceptions.Timeout:
        with stat_lock:
            failed_requests_count += 1
        log_message(f"Worker {worker_id}: EXCEPTION | Error: Request timed out after {REQUEST_TIMEOUT}s", error=True)
    except requests.exceptions.RequestException as e:
        with stat_lock:
            failed_requests_count += 1
        log_message(f"Worker {worker_id}: EXCEPTION | Error: {str(e)}", error=True)
    except Exception as e: # Catch any other unexpected errors
        with stat_lock:
            failed_requests_count += 1
        log_message(f"Worker {worker_id}: UNEXPECTED_EXCEPTION | Error: {str(e)}", error=True)

# --- Main Test Function ---
def run_benchmark():
    global metrics_successful_requests, failed_requests_count
    global first_token_latencies_list, generation_durations_list, actual_output_tokens_list
    global request_log_file_handle, API_URL, MODEL_NAME, TOTAL_REQUESTS_PER_SETTING
    global CONCURRENCY_LEVELS, INPUT_TOKEN_LENGTHS, FIXED_OUTPUT_TOKENS, REQUEST_TIMEOUT

    # Override config with environment variables if set
    API_URL = os.getenv("LLM_API_URL", API_URL)
    MODEL_NAME = os.getenv("LLM_MODEL_NAME", MODEL_NAME)
    TOTAL_REQUESTS_PER_SETTING = int(os.getenv("LLM_TOTAL_REQUESTS", str(TOTAL_REQUESTS_PER_SETTING)))
    CONCURRENCY_LEVELS = [int(c.strip()) for c in os.getenv("LLM_CONCURRENCY_LEVELS", ",".join(map(str, CONCURRENCY_LEVELS))).split(',')]
    INPUT_TOKEN_LENGTHS = [int(t.strip()) for t in os.getenv("LLM_INPUT_TOKEN_LENGTHS", ",".join(map(str, INPUT_TOKEN_LENGTHS))).split(',')]
    # FIXED_OUTPUT_TOKENS is a constant in this script, but could be env var too
    REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", str(REQUEST_TIMEOUT)))

    log_message("Starting LLM Benchmark...", console_only=True)
    log_message(f"API URL: {API_URL}", console_only=True)
    log_message(f"Model Name: {MODEL_NAME}", console_only=True)
    log_message(f"Concurrency Levels: {CONCURRENCY_LEVELS}", console_only=True)
    log_message(f"Input Token Lengths: {INPUT_TOKEN_LENGTHS}", console_only=True)
    log_message(f"Fixed Output Tokens: {FIXED_OUTPUT_TOKENS}", console_only=True)
    log_message(f"Requests per Setting: {TOTAL_REQUESTS_PER_SETTING}", console_only=True)
    log_message(f"Request Timeout: {REQUEST_TIMEOUT}s", console_only=True)

    # Create a timestamped directory for results
    results_dir_name = f"llm_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir_name, exist_ok=True)
    
    detailed_log_path = os.path.join(results_dir_name, "detailed_request_log.txt")
    summary_results_path = os.path.join(results_dir_name, "summary_results.md")

    try:
        request_log_file_handle = open(detailed_log_path, "w", encoding='utf-8')
        
        with open(summary_results_path, "w", encoding='utf-8') as md_file:
            md_file.write(f"# LLM Benchmark Results ({MODEL_NAME})\\n\\n")
            md_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            md_file.write(f"Target Output Tokens: {FIXED_OUTPUT_TOKENS}\\n")
            md_file.write(f"Total Attempts per Setting: {TOTAL_REQUESTS_PER_SETTING}\\n\\n")
            md_file.write("| Concurrency | Input Tokens | Avg First Token Latency (s) | Avg Token Output Rate (tokens/s) | QPS (req/s) | Successful (metrics) | Failed / Invalid |\\n")
            md_file.write("|-------------|--------------|-----------------------------|------------------------------------|-------------|----------------------|------------------|\\n")

            for concurrency in CONCURRENCY_LEVELS:
                for input_tokens in INPUT_TOKEN_LENGTHS:
                    log_message(f"\\n--- Test: Concurrency={concurrency}, InputTokens={input_tokens} ---")

                    metrics_successful_requests = 0
                    failed_requests_count = 0
                    first_token_latencies_list = []
                    generation_durations_list = []
                    actual_output_tokens_list = []
                    
                    prompt_for_test = generate_prompt(input_tokens)
                    overall_test_start_time = time.time()

                    with ThreadPoolExecutor(max_workers=concurrency) as executor:
                        futures = [executor.submit(worker, i + 1, prompt_for_test) for i in range(TOTAL_REQUESTS_PER_SETTING)]
                        for future in futures:
                            try:
                                future.result() # Wait for task to finish and catch exceptions from worker
                            except Exception as e:
                                log_message(f"MainLoop: Worker task ended with unhandled exception: {e}", error=True)
                
                    overall_test_end_time = time.time()
                    total_test_duration = overall_test_end_time - overall_test_start_time

                    avg_ftl = sum(first_token_latencies_list) / len(first_token_latencies_list) if first_token_latencies_list else 0
                    
                    total_generated_tokens = sum(actual_output_tokens_list)
                    total_generation_time_for_tokens = sum(generation_durations_list)
                    
                    avg_token_output_rate = total_generated_tokens / total_generation_time_for_tokens if total_generation_time_for_tokens > 0 else 0
                    
                    # QPS based on requests that yielded valid metrics
                    qps = metrics_successful_requests / total_test_duration if total_test_duration > 0 else 0

                    log_message(f"--- Results: Concurrency={concurrency}, InputTokens={input_tokens} ---")
                    log_message(f"Successful (metrics): {metrics_successful_requests}, Failed/Invalid: {failed_requests_count}")
                    log_message(f"Avg First Token Latency: {avg_ftl:.4f} s")
                    log_message(f"Avg Token Output Rate: {avg_token_output_rate:.2f} tokens/s (based on {total_generated_tokens} tokens over {total_generation_time_for_tokens:.2f}s of generation)")
                    log_message(f"QPS (metrics-based): {qps:.2f} req/s")
                    log_message(f"Total time for this setting: {total_test_duration:.2f}s")

                    md_file.write(f"| {concurrency} | {input_tokens} | {avg_ftl:.4f} | {avg_token_output_rate:.2f} | {qps:.2f} | {metrics_successful_requests} | {failed_requests_count} |\\n")
                    md_file.flush()
    
    finally:
        if request_log_file_handle:
            log_message(f"Detailed log saved to: {os.path.abspath(detailed_log_path)}", console_only=True)
            request_log_file_handle.close()
        log_message(f"Summary results saved to: {os.path.abspath(summary_results_path)}", console_only=True)
        log_message("LLM Benchmark Finished.", console_only=True)

# --- Entry Point ---
if __name__ == "__main__":
    run_benchmark()
