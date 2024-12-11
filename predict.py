from fastapi import FastAPI, File, UploadFile
from mangum import Mangum
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import json

app = FastAPI()

# Load model and preprocess setup
model = torch.load("freshness.pt", map_location=torch.device("cpu"))
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Logic to handle image prediction
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        processed_image = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(processed_image)
        
        # Example of predicting freshness score
        freshness_score = prediction.item()
        return {"freshness_score": freshness_score}

    except Exception as e:
        return {"error": str(e)}

# Lambda handler function (this is the actual entry point for AWS Lambda)
lambda_handler = Mangum(app)  # Mangum will convert FastAPI app into a Lambda handler
