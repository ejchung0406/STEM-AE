import torch
from transformers import AutoImageProcessor, ResNetForImageClassification
from datasets import load_dataset, concatenate_datasets
import PIL
from tqdm import tqdm
import numpy as np

# Load the dataset with trust_remote_code=True to avoid FutureWarning
dataset = load_dataset("frgfm/imagenette", "full_size", trust_remote_code=True)

# Load the AutoImageProcessor and ResNetForImageClassification
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Iterate over the dataset and perform inference on only 2 images
d = concatenate_datasets([dataset["train"], dataset["validation"]])
for i, example in enumerate(tqdm(d, total=2, desc="Processing images")):
    if i >= 2:  # Only process first 2 images
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

    # Preprocess the image
    inputs = image_processor(image_np, return_tensors="pt")
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
