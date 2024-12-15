from fastapi import FastAPI, File, UploadFile
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

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
    try:
        # Read and process the uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        processed_image = transform(image).unsqueeze(0)

        # Predict freshness score
        with torch.no_grad():
            prediction = model(processed_image)

        freshness_score = prediction.item()
        return {"freshness_score": freshness_score}

    except Exception as e:
        return {"error": str(e)}
