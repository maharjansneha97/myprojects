"""
Custom Scene Classifier - ResNet-18 Architecture
Built from scratch without pretrained weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class CustomResNet18(nn.Module):
    """
    Custom ResNet-18 for Scene Classification
    Built from scratch - NO PRETRAINED WEIGHTS
    """
    
    def __init__(self, num_classes=30):
        super(CustomResNet18, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, out_channels, blocks, stride=1):
        """
        Create a residual layer
        """
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        Initialize weights using He initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling and classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class CustomSceneClassifier(nn.Module):
    """
    Wrapper class for the custom scene classifier
    Provides additional functionality for inference
    """
    
    def __init__(self, num_classes=30):
        super(CustomSceneClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model = CustomResNet18(num_classes=num_classes)
        
        print(f"✓ Custom ResNet-18 Scene Classifier initialized")
        print(f"  • Number of classes: {num_classes}")
        print(f"  • Total parameters: {self.count_parameters():,}")
        print(f"  • NO PRETRAINED WEIGHTS - Training from scratch")
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x, top_k=5):
        """
        Make predictions with confidence scores
        
        Args:
            x: Input tensor [batch_size, 3, 224, 224]
            top_k: Number of top predictions to return
        
        Returns:
            predictions: Top-k class indices
            confidences: Top-k confidence scores
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, top_k, dim=1)
        
        return top_indices, top_probs


def test_model():
    """Test the custom scene classifier"""
    print("\n" + "="*60)
    print("TESTING CUSTOM SCENE CLASSIFIER")
    print("="*60)
    
    # Create model
    model = CustomSceneClassifier(num_classes=15)
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\n📊 Testing forward pass...")
    print(f"  Input shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: ({batch_size}, 15)")
    
    # Test prediction
    print(f"\n🔍 Testing prediction...")
    top_indices, top_probs = model.predict(dummy_input, top_k=5)
    print(f"  Top-5 indices shape: {top_indices.shape}")
    print(f"  Top-5 probabilities shape: {top_probs.shape}")
    print(f"  Sample top-5 classes: {top_indices[0].tolist()}")
    print(f"  Sample top-5 probs: {top_probs[0].tolist()}")
    
    print("\n✓ Model test passed!")
    
    return model


if __name__ == "__main__":
    test_model()
