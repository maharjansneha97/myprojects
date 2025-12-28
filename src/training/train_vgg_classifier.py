"""
Training Script for Custom VGG Scene Classifier
Trains the custom VGG model on scene classification dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import argparse
from tqdm import tqdm
import json

# Import custom VGG model
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.custom_vgg import CustomVGG


def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    """
    Create data loaders for training and validation
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for training
        num_workers: Number of worker processes
        
    Returns:
        train_loader, val_loader, class_names
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=Path(data_dir) / 'train',
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=Path(data_dir) / 'val',
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    class_names = train_dataset.classes
    
    return train_loader, val_loader, class_names


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch
    
    Returns:
        Average loss, accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Validate the model
    
    Returns:
        Average loss, accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def main(args):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\nLoading dataset...")
    train_loader, val_loader, class_names = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    num_classes = len(class_names)
    print(f"✓ Dataset loaded")
    print(f"  Classes: {num_classes}")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = CustomVGG(num_classes=num_classes, dropout=args.dropout)
    model = model.to(device)
    print(f"✓ Model created")
    print(f"  Parameters: {model.get_num_parameters():,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step,
        gamma=args.lr_gamma
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)
    
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  ✓ New best validation accuracy! Saving model...")
            
            # Save model
            save_path = Path(args.save_dir) / 'custom_vgg.pth'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(save_path))
            
            # Save class names
            classes_path = Path(args.save_dir) / 'classes.txt'
            with open(classes_path, 'w') as f:
                for class_name in class_names:
                    f.write(f"{class_name}\n")
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {save_path}")
    
    # Save training history
    history_path = Path(args.save_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Custom VGG Scene Classifier')
    
    # Dataset parameters
    parser.add_argument('--data-dir', type=str, default='dataset/scenes',
                       help='Path to dataset directory')
    parser.add_argument('--save-dir', type=str, default='models/scene_classifier',
                       help='Directory to save model')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout probability')
    
    # Learning rate scheduler
    parser.add_argument('--lr-step', type=int, default=30,
                       help='Learning rate decay step')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                       help='Learning rate decay factor')
    
    # Other parameters
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    main(args)
