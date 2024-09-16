from fastapi import FastAPI, File, UploadFile
from typing import List
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
import io

app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = models.efficientnet_b1(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 4)
model.load_state_dict(torch.load('final_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Define the image transformations
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define label mapping
label_mapping = {
    0: 'Smoke',
    1: 'Fire',
    2: 'None',
    3: 'Smoke and Fire'
}

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return test_transform(image).unsqueeze(0)

@app.post("/predict/")
async def predict(files: List[UploadFile] = File(...)):
    predictions = []
    
    for file in files:
        image_bytes = await file.read()
        tensor = transform_image(image_bytes).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs, 1)
        
        # Map label to string
        predicted_label = label_mapping.get(predicted.item(), "Unknown")
        predictions.append({"file_name": file.filename, "class": predicted_label})
    
    return {"predictions": predictions}
