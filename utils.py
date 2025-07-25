# Pre-processing and Grad-CAM utilities
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from config import NORMALIZE_MEAN, NORMALIZE_STD, IMAGE_SIZE, DEVICE

def load_model(model_path, num_classes=7):
    """Load trained model with proper error handling"""
    try:
        # Import here to avoid circular imports
        from models.cnn_model import create_resnet18_model
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model = create_resnet18_model(num_classes=num_classes)

        model.load_state_dict(torch.load('models/saved_model.pth', map_location='cpu'))
        model.eval()    
        return model

    except ImportError:
        raise ImportError("Model architecture not found. Create models/cnn_model.py first.")
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    return transform(image).unsqueeze(0)

class GradCAM:
    def __init__(self, model, target_layer_name='layer4'):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        
    def generate_cam(self, input_tensor, class_idx):
        # Find target layer
        target_layer = dict(self.model.named_modules())[self.target_layer_name]
        
        # Register hooks
        h1 = target_layer.register_forward_hook(self.save_activation)
        h2 = target_layer.register_backward_hook(self.save_gradient)
        self.hooks = [h1, h2]
        
        try:
            # Forward pass
            output = self.model(input_tensor)
            
            # Backward pass
            self.model.zero_grad()
            output[0][class_idx].backward()
            
            # Generate CAM
            weights = torch.mean(self.gradients, dim=[2, 3])
            cam = torch.zeros(self.activations.shape[2:])
            
            for i, w in enumerate(weights[0]):
                cam += w * self.activations[0][i]
            
            cam = F.relu(cam)
            cam = cam / torch.max(cam) if torch.max(cam) > 0 else cam
            return cam.detach().numpy()
            
        finally:
            # Clean up hooks
            for hook in self.hooks:
                hook.remove()


def get_gradcam(model, input_tensor, class_idx):
    """Generate GradCAM visualization"""
    # Get the last convolutional layer
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    
    if target_layer is None:
        return None
    
    # Generate GradCAM
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(input_tensor, class_idx)
    
    # Resize CAM to input size
    cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return heatmap

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
