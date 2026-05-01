import ollama
import time

start_time = time.time()

response = ollama.chat(
    model="qwen2.5:7b-instruct",  # Try qwen3:32b if resources are available
    messages=[
        {"role": "user", "content": "Summarize the theory of evolution."}
    ]
)

print(response["message"]["content"])

end_time = time.time()

elapsed = end_time - start_time
minutes, seconds = divmod(elapsed, 60)

print("-----")
if minutes >= 1:
    print(f"Response time: {int(minutes)} min {seconds:.2f} sec")
else:
    print(f"Response time: {seconds:.2f} sec")