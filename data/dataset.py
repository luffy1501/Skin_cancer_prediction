# Torch Dataset
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from config import CLASS_NAMES, IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD

class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        
        # Create label mapping
        self.label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, f"{row['image_id']}.jpg")
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.label_map[row['dx']]
        
        return image, label

def get_transforms(training=True):
    """Get data transforms for training and validation"""
    if training:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])

def create_data_loaders(metadata_path, image_dirs, batch_size=32, test_size=0.2):
    """Create train and validation data loaders"""
    
    # Load metadata
    df = pd.read_csv(metadata_path)
    
    # Combine image directories
    all_images = []
    for img_dir in image_dirs:
        for img_file in os.listdir(img_dir):
            if img_file.endswith('.jpg'):
                img_id = img_file.replace('.jpg', '')
                if img_id in df['image_id'].values:
                    all_images.append((img_id, img_dir))
    
    # Filter dataframe for available images
    available_ids = [img[0] for img in all_images]
    df_filtered = df[df['image_id'].isin(available_ids)]
    
    # Split data
    train_df, val_df = train_test_split(
        df_filtered, 
        test_size=test_size, 
        stratify=df_filtered['dx'], 
        random_state=42
    )
    
    # Create image directory mapping
    img_dir_map = {img[0]: img[1] for img in all_images}
    
    # Create datasets
    train_dataset = SkinLesionDataset(
        train_df, 
        img_dir_map, 
        transform=get_transforms(training=True)
    )
    
    val_dataset = SkinLesionDataset(
        val_df, 
        img_dir_map, 
        transform=get_transforms(training=False)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    return train_loader, val_loader
