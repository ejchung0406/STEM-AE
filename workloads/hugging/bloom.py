# Use a pipeline as a high-level helper
import torch
from transformers import pipeline, set_seed
from tqdm import tqdm

pipe = pipeline("text-generation", model="ybelkada/bloom-1b7-8bit")

# Set the seed
set_seed(42)

# Generate text on the GPU
for i in tqdm(range(100)):
  results = pipe("Hello, I'm a language model,", max_length=100, truncation=True)
