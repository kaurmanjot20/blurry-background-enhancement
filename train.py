import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
from camera_pipeline.deblur_cnn import MobileNetDeblur
import argparse

# --- Dataset ---
class DeblurDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.sharp_dir = os.path.join(root_dir, 'sharp')
        self.blur_dir = os.path.join(root_dir, 'blur')
        self.transform = transform
        
        self.image_files = [f for f in os.listdir(self.sharp_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Split train/val
        split_idx = int(len(self.image_files) * 0.9)
        if mode == 'train':
            self.image_files = self.image_files[:split_idx]
        else:
            self.image_files = self.image_files[split_idx:]
            
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        fname = self.image_files[idx]
        
        sharp_path = os.path.join(self.sharp_dir, fname)
        blur_path = os.path.join(self.blur_dir, fname)
        
        sharp_img = Image.open(sharp_path).convert('RGB')
        blur_img = Image.open(blur_path).convert('RGB')
        
        if self.transform:
            # We need to apply same random crop to both
            # Manually handle transforms to ensure sync
            
            # Resize to reasonable Training Size (e.g., 256x256)
            # For simplicity, let's just resize
            sharp_img = sharp_img.resize((256, 256))
            blur_img = blur_img.resize((256, 256))
            
            sharp_tensor = transforms.ToTensor()(sharp_img)
            blur_tensor = transforms.ToTensor()(blur_img)
            
            return blur_tensor, sharp_tensor

        return blur_img, sharp_img

# --- SSIM Loss ---
# Simple implementation or use Library
# For simplicity, we stick to L1 Loss + Perceptual (VGG) is best, but L1 is good start.
# Let's verify imports first.

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Data
    train_dataset = DeblurDataset(args.data, mode='train', transform=True)
    val_dataset = DeblurDataset(args.data, mode='val', transform=True)
    
    if len(train_dataset) == 0:
        print("Error: No images found. Did you run create_dataset.py?")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) # workers=0 for windows safe
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Model
    model = MobileNetDeblur(pretrained=False).to(device)
    model.train() # Set to train mode
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss() # Pixel-wise distance
    
    best_loss = float('inf')
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for blur, sharp in pbar:
            blur, sharp = blur.to(device), sharp.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            output = model(blur)
            
            # Loss
            loss = criterion(output, sharp)
            
            # Backprop
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for blur, sharp in val_loader:
                blur, sharp = blur.to(device), sharp.to(device)
                output = model(blur)
                loss = criterion(output, sharp)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Results: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save Best
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved Best Model!")
            
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset", help="Path to dataset folder detected by create_dataset.py")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    train(args)
