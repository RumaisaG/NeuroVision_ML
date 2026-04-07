# predict.py  —  FastAPI ML Service for NeuroVision
# Run: uvicorn predict:app --host 0.0.0.0 --port 8000

import io, base64, torch, numpy as np
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from torchvision import transforms
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn as nn
from pathlib import Path
import cv2
from pytorch_grad_cam import GradCAMPlusPlus
import httpx

from torchvision.models import densenet169

class AlzheimerClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout):
        super().__init__()
        self.model_name = model_name
        if model_name == "densenet169":
            self.backbone = densenet169(weights=None)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, 512),
            nn.ReLU(inplace=True), nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.75), nn.Linear(512, 256),
            nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))

# Load model 
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "densenet169_fastapi.pt"
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device)
model = AlzheimerClassifier(
    checkpoint["model_name"],
    checkpoint["num_classes"],
    checkpoint["dropout"]
).to(device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

CLASS_NAMES = checkpoint["class_names"]
IMG_SIZE    = checkpoint["img_size"]
MEAN        = checkpoint["imagenet_mean"]
STD         = checkpoint["imagenet_std"]

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

def get_target_layer():
    if checkpoint["model_name"] == "densenet169":
        return model.backbone.features.denseblock4.denselayer32.conv2
    return model.backbone.layer4[-1].conv3

app = FastAPI(title="NeuroVision ML Service")

@app.get("/health")
def health():
    return {"status": "ok", "model": checkpoint["model_name"]}

@app.post("/predict")
async def predict(payload: dict):
    image_url = payload["image_url"]
    gradcam_enabled = payload.get("gradcam_enabled", True)

    # Load image 
   # Fetch image from Supabase
    async with httpx.AsyncClient() as client:
        response = await client.get(image_url)

        if response.status_code != 200:
            return JSONResponse(
                status_code=400,
                content={"error": "Failed to download image"}
            )
    
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    orig_arr = np.array(img).astype(np.float32) / 255.0

    #  Improved brain mask 
    gray = np.mean(orig_arr, axis=2)

    brain_mask = (gray > 0.05).astype(np.uint8)

    kernel = np.ones((5,5), np.uint8)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

    brain_mask = cv2.GaussianBlur(brain_mask.astype(np.float32), (5,5), 0)
    brain_mask = (brain_mask > 0.2).astype(np.float32)

    #  Preprocess image 
    img_tensor = transforms.ToTensor()(img)
    img_tensor = transforms.Normalize(MEAN, STD)(img_tensor)

    mask_tensor = torch.tensor(brain_mask).unsqueeze(0)
    img_tensor = img_tensor * mask_tensor

    input_tensor = img_tensor.unsqueeze(0).to(device)

    #  Prediction 
    with torch.no_grad():
        logits = model(input_tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    # Grad-CAM++ 
    target_layer = get_target_layer()

    if gradcam_enabled:

        target_layer = get_target_layer()

        with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as cam:
            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        grayscale_cam = grayscale_cam * brain_mask
        grayscale_cam = cv2.GaussianBlur(grayscale_cam, (9,9), 0)

        if grayscale_cam.max() > 0:
            grayscale_cam = grayscale_cam / grayscale_cam.max()

        heatmap_img = show_cam_on_image(orig_arr, grayscale_cam, use_rgb=True)

        heatmap_pil = Image.fromarray(heatmap_img)
        buf = io.BytesIO()
        heatmap_pil.save(buf, format="PNG")

        gradcam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # ── API Response 
    return JSONResponse({
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probs[i]), 4)
            for i in range(len(CLASS_NAMES))
        },
        "gradcam_base64": gradcam_b64
    })