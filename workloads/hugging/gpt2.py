import torch
from transformers import pipeline, set_seed
from tqdm import tqdm

# Move the model to the GPU
generator = pipeline('text-generation', model='gpt2', device=0)

# Set the seed
set_seed(42)

# Generate text on the GPU
for i in tqdm(range(1000)):
  results = generator("Hello, I'm a language model,", max_length=100, num_return_sequences=100, truncation=True)

# Move the generated text to the CPU for printing
# for i in range(len(results)):
#   print(results[i]['generated_text'])

