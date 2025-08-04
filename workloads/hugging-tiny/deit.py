from transformers import AutoFeatureExtractor, DeiTForImageClassificationWithTeacher
from PIL import Image
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import requests
import torch
import numpy as np
import random

feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/deit-small-distilled-patch16-224')
model = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-small-distilled-patch16-224')

# Load the dataset with trust_remote_code=True to avoid FutureWarning
dataset = load_dataset("frgfm/imagenette", "full_size", trust_remote_code=True)
shuffled_dataset = dataset.shuffle(seed=42)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Iterate over the dataset and perform inference on only 4 images
d = concatenate_datasets([dataset["train"], dataset["validation"]])
for i, example in enumerate(tqdm(d, total=2, desc="Processing images")):
  if i >= 2:  # Only process first 4 images
    break
  image = example["image"]
  
  # Convert image to numpy array
  image_np = np.array(image)

  # Check if the image has three dimensions
  if len(image_np.shape) == 3:
    # If it has three dimensions, select the first three channels
    image_np = image_np[:, :, :3]
  elif len(image_np.shape) == 2:
    # If it has two dimensions (grayscale), convert to RGB
    image_np = np.stack((image_np,) * 3, axis=-1)

  # inputs = image_processor(image_np, return_tensors="pt")

  # Preprocess the image
  inputs = feature_extractor(images=image_np, return_tensors="pt")
  inputs.to(device)

  # Perform inference
  with torch.no_grad():
    logits = model(**inputs).logits
  
  # Move logits back to CPU for processing
  logits = logits.cpu()

  # Determine the predicted label
  predicted_label = logits.argmax(-1).item()
  predicted_class = model.config.id2label[predicted_label]

  # Print the predicted class for each image
  print(f"Image {i+1}: Predicted Class: {predicted_class}")

print(f"Finished inference of 2 images")