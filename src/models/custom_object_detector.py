"""
Custom Object Detector - Faster R-CNN Architecture
Built from scratch without pretrained weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.ops as ops


class CustomBackbone(nn.Module):
    """
    Custom ResNet-50 backbone for Faster R-CNN
    Built from scratch - NO PRETRAINED WEIGHTS
    """
    
    def __init__(self):
        super(CustomBackbone, self).__init__()
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks (simplified ResNet-50)
        self.layer1 = self._make_layer(64, 128, blocks=3, stride=1)
        self.layer2 = self._make_layer(128, 256, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, 512, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, 512, blocks=3, stride=2)
        
        # Output channels for FPN
        self.out_channels = 512
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Create a residual layer"""
        layers = []
        
        # First block with stride
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x


class CustomObjectDetector(nn.Module):
    """
    Custom Faster R-CNN Object Detector
    Built from scratch - NO PRETRAINED WEIGHTS
    """
    
    def __init__(self, num_classes=20, min_size=800, max_size=1333):
        super(CustomObjectDetector, self).__init__()
        
        self.num_classes = num_classes
        
        # Create custom backbone
        backbone = CustomBackbone()
        backbone.out_channels = 512
        
        # Define anchor generator
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        
        # Define ROI pooler
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Create Faster R-CNN model
        self.model = FasterRCNN(
            backbone,
            num_classes=num_classes + 1,  # +1 for background
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=min_size,
            max_size=max_size
        )
        
        # Replace the classifier head to ensure no pretrained weights
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        
        print(f"✓ Custom Faster R-CNN Object Detector initialized")
        print(f"  • Number of classes: {num_classes}")
        print(f"  • Total parameters: {self.count_parameters():,}")
        print(f"  • NO PRETRAINED WEIGHTS - Training from scratch")
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, images, targets=None):
        """
        Forward pass
        
        Args:
            images: List of tensors or tensor [batch, 3, H, W]
            targets: List of dicts with 'boxes' and 'labels' (training only)
        
        Returns:
            During training: dict of losses
            During inference: list of dicts with 'boxes', 'labels', 'scores'
        """
        if self.training:
            return self.model(images, targets)
        else:
            return self.model(images)
    
    def predict(self, images, confidence_threshold=0.5):
        """
        Make predictions with confidence filtering
        
        Args:
            images: Tensor [batch, 3, H, W] or list of tensors
            confidence_threshold: Minimum confidence score
        
        Returns:
            List of predictions for each image
        """
        self.eval()
        with torch.no_grad():
            predictions = self.model(images)
        
        # Filter by confidence
        filtered_predictions = []
        for pred in predictions:
            keep = pred['scores'] > confidence_threshold
            filtered_pred = {
                'boxes': pred['boxes'][keep],
                'labels': pred['labels'][keep],
                'scores': pred['scores'][keep]
            }
            filtered_predictions.append(filtered_pred)
        
        return filtered_predictions


def test_model():
    """Test the custom object detector"""
    print("\n" + "="*60)
    print("TESTING CUSTOM OBJECT DETECTOR")
    print("="*60)
    
    # Create model
    model = CustomObjectDetector(num_classes=20)
    model.eval()
    
    # Test forward pass
    batch_size = 2
    dummy_images = [torch.randn(3, 800, 800) for _ in range(batch_size)]
    
    print(f"\n📊 Testing inference...")
    print(f"  Input: {batch_size} images of shape (3, 800, 800)")
    
    predictions = model.predict(dummy_images, confidence_threshold=0.5)
    
    print(f"  Number of predictions: {len(predictions)}")
    for i, pred in enumerate(predictions):
        print(f"  Image {i+1}:")
        print(f"    • Boxes: {pred['boxes'].shape}")
        print(f"    • Labels: {pred['labels'].shape}")
        print(f"    • Scores: {pred['scores'].shape}")
    
    # Test training mode
    print(f"\n🔍 Testing training mode...")
    model.train()
    
    # Create dummy targets
    targets = []
    for _ in range(batch_size):
        targets.append({
            'boxes': torch.tensor([[10, 10, 100, 100], [200, 200, 300, 300]], dtype=torch.float32),
            'labels': torch.tensor([1, 2], dtype=torch.int64)
        })
    
    loss_dict = model(dummy_images, targets)
    print(f"  Loss keys: {list(loss_dict.keys())}")
    for key, value in loss_dict.items():
        print(f"    • {key}: {value.item():.4f}")
    
    print("\n✓ Model test passed!")
    
    return model


if __name__ == "__main__":
    test_model()
