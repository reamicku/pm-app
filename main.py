import torch
from transformers import pipeline
import time

def print_with_time(message):
    """Prints a message with the time elapsed since the start of the script."""
    elapsed_time = time.time() - start_time
    print(f"[{elapsed_time:.2f}]\t{message}")
start_time = time.time()



print_with_time("=== LOAD MODEL")
pipe = pipeline(model="unsloth/Llama-3.2-1B-Instruct", device_map="auto")

print_with_time("=== PREPARE PIPELINE")
output = pipe("How do Large Language Models work?", do_sample=True, top_p=0.95, max_new_tokens=500)

print_with_time("=== START INFERENCE")
print_with_time(f"Output: {output}")
