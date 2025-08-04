from transformers import BloomConfig, BloomForCausalLM, BloomTokenizerFast
import torch

# Use BLOOM tokenizer
tokenizer = BloomTokenizerFast.from_pretrained('bigscience/bloom-560m')

# Create BLOOM config with similar workload size
config = BloomConfig.from_pretrained('bigscience/bloom-560m')
config.n_layer = 2  # Similar to BERT-base and GPT-2 standard (default BLOOM-560m has 24 layers)
config.hidden_size = 768  # Similar to BERT-base hidden size (default BLOOM-560m has 1024)
config.n_head = 12  # Adjust attention heads accordingly

# Create BLOOM model for causal language modeling
model = BloomForCausalLM(config)
model.to("cuda")

# Test with text generation
input_text = "Hello world"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

# Generate text with the model
with torch.no_grad():
    outputs = model.generate(
        input_ids, 
        max_length=5,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")
