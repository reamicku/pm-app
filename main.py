import torch
from transformers import pipeline

pipe = pipeline(model="unsloth/Llama-3.2-1B-Instruct", device_map="auto")

output = pipe("This is a cool example!", do_sample=True, top_p=0.95)
