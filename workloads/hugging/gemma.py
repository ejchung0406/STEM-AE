# https://huggingface.co/google/gemma-2b
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")

input_text = "Hello, I'm a language model,"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

for i in tqdm(range(1000)):
  set_seed(i)
  outputs = model.generate(**input_ids)
  # print(tokenizer.decode(outputs[0]))