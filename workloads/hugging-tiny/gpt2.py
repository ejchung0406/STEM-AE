from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GPT2Config.from_pretrained('gpt2')
config.num_hidden_layers = 4 

model = GPT2LMHeadModel(config)

model.to("cuda")

input_text = "Hello"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
outputs = model.generate(input_ids, max_length=5)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
