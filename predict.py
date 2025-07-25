import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import argparse
from models.cnn_model import ResNetClassifier
from config import CLASS_NAMES, MODEL_PATH, NORMALIZE_MEAN, NORMALIZE_STD, IMAGE_SIZE

def load_model(model_path):
    """Load trained model for inference"""
    model = ResNetClassifier(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image_path):
    """Preprocess image for model input"""
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    
    return transform(image).unsqueeze(0)

def predict_single_image(model, image_tensor):
    """Make prediction on a single image"""
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = probabilities.argmax().item()
        confidence = probabilities[predicted_class].item()
    
    return predicted_class, confidence, probabilities

def main():
    parser = argparse.ArgumentParser(description='Predict skin lesion class')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to trained model')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = load_model(args.model)
    
    # Preprocess image
    print(f"Processing image: {args.image}")
    image_tensor = preprocess_image(args.image)
    
    # Make prediction
    pred_class, confidence, probabilities = predict_single_image(model, image_tensor)
    
    # Display results
    print(f"\nPrediction Results:")
    print(f"Predicted Class: {CLASS_NAMES[pred_class]}")
    print(f"Confidence: {confidence:.2%}")
    print(f"\nAll Class Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"  {CLASS_NAMES[i]}: {prob:.2%}")

if __name__ == "__main__":
    main()
