import torch
from transformers import pipeline
import time

def print_with_time(message):
    """Prints a message with the time elapsed since the start of the script."""
    elapsed_time = time.time() - start_time
    print(f"[{elapsed_time:.2f}]\t{message}")
start_time = time.time()



MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
print_with_time(f"Loading model {MODEL_NAME}")
pipe = pipeline(model=MODEL_NAME, device_map="auto")

print_with_time("Prepare pipeline")
output = pipe("How do Large Language Models work?", do_sample=True, top_p=0.95, max_new_tokens=500)

print_with_time("Begin inference")
print_with_time(f"Output: {output}")
