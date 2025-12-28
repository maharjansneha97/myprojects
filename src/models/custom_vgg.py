"""
Custom VGG-style CNN for Scene Classification
Built from scratch without pretrained weights

Academic Justification:
- VGG Architecture: Simple yet effective deep CNN design
- Stacked 3x3 convolutions: Proven effective for feature extraction
- Batch Normalization: Improves training stability
- Dropout: Prevents overfitting
- Fully custom: Complete control over architecture
"""

import torch
import torch.nn as nn


class CustomVGG(nn.Module):
    """
    Custom VGG-style Convolutional Neural Network
    
    Architecture:
    - 5 convolutional blocks with increasing filters (64, 128, 256, 512, 512)
    - Each block has 2-3 conv layers followed by max pooling
    - 3 fully connected layers with dropout
    - Output: 10 scene classes
    
    Parameters: ~15M
    Input size: 224x224x3
    """
    
    def __init__(self, num_classes=10, dropout=0.5):
        """
        Initialize Custom VGG model
        
        Args:
            num_classes: Number of output classes (default: 10)
            dropout: Dropout probability (default: 0.5)
        """
        super(CustomVGG, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional Block 1: 64 filters
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112
        )
        
        # Convolutional Block 2: 128 filters
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56
        )
        
        # Convolutional Block 3: 256 filters
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 56 -> 28
        )
        
        # Convolutional Block 4: 512 filters
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28 -> 14
        )
        
        # Convolutional Block 5: 512 filters
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14 -> 7
        )
        
        # Adaptive pooling to handle varying input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Fully Connected Layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, 3, 224, 224)
            
        Returns:
            Output logits (batch_size, num_classes)
        """
        # Convolutional blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        """
        Initialize model weights using He initialization for ReLU
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def get_num_parameters(self):
        """
        Get total number of parameters
        
        Returns:
            int: Total parameters
        """
        return sum(p.numel() for p in self.parameters())


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Custom VGG-style CNN for Scene Classification")
    print("=" * 70)
    
    # Create model
    model = CustomVGG(num_classes=10)
    
    # Model summary
    print(f"\n✓ Model created successfully")
    print(f"✓ Number of classes: {model.num_classes}")
    print(f"✓ Total parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    print(f"\n{'Testing forward pass...'}")
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"✓ Input shape: {dummy_input.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Output logits sample: {output[0][:5].detach().numpy()}")
    
    # Architecture breakdown
    print(f"\n{'Architecture Breakdown:'}")
    print(f"  Conv Block 1: 3 → 64 → 64 (MaxPool)")
    print(f"  Conv Block 2: 64 → 128 → 128 (MaxPool)")
    print(f"  Conv Block 3: 128 → 256 → 256 → 256 (MaxPool)")
    print(f"  Conv Block 4: 256 → 512 → 512 → 512 (MaxPool)")
    print(f"  Conv Block 5: 512 → 512 → 512 → 512 (MaxPool)")
    print(f"  FC Layers: 25088 → 4096 → 4096 → 10")
    
    print("\n" + "=" * 70)
