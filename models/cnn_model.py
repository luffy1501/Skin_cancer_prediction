import torch
import torch.nn as nn
from torchvision import models
import os

def create_resnet18_model(num_classes=7):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def load_model(weights_path, num_classes=7):
    print(f"🔍 Attempting to load model from: {weights_path}")
    model = create_resnet18_model(num_classes)
    print("✅ Model architecture created.")
    
    state_dict = torch.load(weights_path, map_location="cpu")
    print("📦 Loaded state_dict keys:", list(state_dict.keys())[:5])
    
    model.load_state_dict(state_dict)
    print("✅ Weights loaded into model.")
    
    model.eval()
    return model
def preprocess_image(image):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension