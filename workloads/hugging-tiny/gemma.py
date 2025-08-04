from transformers import GemmaConfig, GemmaForCausalLM, GemmaTokenizer

tokenizer = GemmaTokenizer.from_pretrained("google/gemma-2b")
config = GemmaConfig.from_pretrained("google/gemma-2b")
config.num_hidden_layers = 2

model = GemmaForCausalLM(config)

model.to("cuda")

input_text = "Hello"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
outputs = model.generate(input_ids, max_length=5)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
