from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = 2  # For binary classification (e.g., sentiment analysis)

model = BertForSequenceClassification(config)
model.to("cuda")

# For classification task instead of generation
input_text = "Hello world this is a test"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
input_ids = inputs.input_ids.to("cuda")
attention_mask = inputs.attention_mask.to("cuda")

# Forward pass for classification
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
