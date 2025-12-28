"""
Training Script for Custom Single-Stage Object Detector
Trains the custom detector model on object detection dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import cv2
import numpy as np
from PIL import Image

# Import custom Detector model and utils
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.custom_detector import CustomDetector
from models.detection_utils import calculate_iou, xywh_to_xyxy, generate_anchors

class CustomDetectionDataset(Dataset):
    """
    Custom Dataset for Object Detection
    Expects images and labels (YOLO format)
    """
    def __init__(self, data_dir, split='train', img_size=416, grid_size=13, num_anchors=3):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.grid_size = grid_size
        self.num_anchors = num_anchors
        
        # Load image paths
        self.img_dir = self.data_dir / split / 'images'
        self.label_dir = self.data_dir / split / 'labels'
        
        # If split dirs don't exist, try looking in root (simplified structure)
        if not self.img_dir.exists():
            self.img_dir = self.data_dir / 'images'
            self.label_dir = self.data_dir / 'labels'
            
        self.img_files = sorted(list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.png')))
        
        # Anchors (should match model)
        self.anchors = torch.tensor([
            [1.3, 1.9], [3.6, 2.8], [2.2, 2.2]
        ])
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_dir / img_path.with_suffix('.txt').name
        
        # Load Image
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        # Resize/Transform
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img)
        
        # Load Labels
        targets = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    # class, x_center, y_center, width, height (normalized)
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx = float(parts[1])
                        cy = float(parts[2])
                        bw = float(parts[3])
                        bh = float(parts[4])
                        targets.append([cls, cx, cy, bw, bh])
        
        targets = torch.tensor(targets)
        
        # Encode targets for grid
        # Output: (grid_size, grid_size, num_anchors, 6) -> [conf, x, y, w, h, class]
        label_matrix = torch.zeros((self.grid_size, self.grid_size, self.num_anchors, 6))
        
        if len(targets) > 0:
            for box in targets:
                cls, cx, cy, bw, bh = box
                
                # Determine grid cell
                i = int(cy * self.grid_size) # y
                j = int(cx * self.grid_size) # x
                
                # Assign to best anchor
                ious = [calculate_iou([0, 0, bw, bh], [0, 0, anchor[0], anchor[1]]) for anchor in self.anchors]
                best_anchor = np.argmax(ious)
                
                # Relative coordinates in cell
                tx = cx * self.grid_size - j
                ty = cy * self.grid_size - i
                
                # Assign to grid
                if i < self.grid_size and j < self.grid_size:
                    # Objectness mask
                    label_matrix[i, j, best_anchor, 0] = 1 
                    # Box coordinates
                    label_matrix[i, j, best_anchor, 1:5] = torch.tensor([tx, ty, bw, bh])
                    # Class label
                    label_matrix[i, j, best_anchor, 5] = cls
                    
        return img_tensor, label_matrix

def custom_loss(predictions, targets, lambda_coord=5.0, lambda_noobj=0.5):
    """
    Custom Loss Function (Simplified YOLO Loss)
    """
    # predictions: (batch, grid, grid, anchors, 5+num_classes)
    # targets: (batch, grid, grid, anchors, 6) -> [conf, x, y, w, h, class]
    
    # Masks
    obj_mask = targets[..., 0] == 1
    noobj_mask = targets[..., 0] == 0
    
    # Coordinate Loss (MSE)
    # Only verify coordinates where object exists
    coord_loss = 0
    if torch.sum(obj_mask) > 0:
        # Box coordinates loss
        # predictions[..., 0:2] are x,y (sigmoid)
        # predictions[..., 2:4] are w,h (exp) -> simplified here to direct regression for stability in custom training
        
        pred_box = predictions[..., 0:4][obj_mask]
        target_box = targets[..., 1:5][obj_mask]
        
        coord_loss = nn.MSELoss()(pred_box, target_box)
        
    # Objectness Loss (BCE/MSE)
    # Confidence score for objects
    obj_conf_loss = 0 
    if torch.sum(obj_mask) > 0:
        pred_obj = predictions[..., 4][obj_mask]
        target_obj = targets[..., 0][obj_mask] # All 1s
        obj_conf_loss = nn.MSELoss()(pred_obj, target_obj)
        
    # Confidence score for no objects
    pred_noobj = predictions[..., 4][noobj_mask]
    target_noobj = targets[..., 0][noobj_mask] # All 0s
    noobj_conf_loss = nn.MSELoss()(pred_noobj, target_noobj)
    
    # Class Loss (Cross Entropy)
    class_loss = 0
    if torch.sum(obj_mask) > 0:
        pred_cls = predictions[..., 5:][obj_mask] # (N_obj, num_classes)
        target_cls = targets[..., 5][obj_mask].long() # (N_obj,)
        class_loss = nn.CrossEntropyLoss()(pred_cls, target_cls)
    
    # Total Loss
    loss = (lambda_coord * coord_loss) + obj_conf_loss + (lambda_noobj * noobj_conf_loss) + class_loss
    return loss, (coord_loss, obj_conf_loss, noobj_conf_loss, class_loss)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize Model
    model = CustomDetector(num_classes=args.num_classes, num_anchors=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Dataset and DataLoader
    print(f"Loading dataset from {args.data_dir}...")
    dataset = CustomDetectionDataset(args.data_dir, img_size=416)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    print(f"Starting training on {device}...")
    print(f"Images: {len(dataset)}")
    model.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for imgs, targets in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            output = model(imgs)
            
            # Loss
            loss, components = custom_loss(output, targets)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
            
    # Save Model
    save_path = Path(args.save_dir) / 'custom_detector.pth'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='dataset/detection')
    parser.add_argument('--save-dir', type=str, default='models/object_detector')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--num-classes', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    train(args)
