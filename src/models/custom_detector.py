"""
Custom CNN-based Single-Stage Object Detector
Built from scratch without pretrained weights

Academic Justification:
- Single-stage detection: Efficient one-pass inference
- Anchor-based: Uses predefined anchor boxes for detection
- Grid-based prediction: Divides image into grid cells
- Multi-scale features: Captures objects at different scales
- Fully custom: Complete control over architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomDetector(nn.Module):
    """
    Custom Single-Stage Object Detector
    
    Architecture:
    - Backbone: CNN feature extractor (5 conv blocks)
    - Detection Head: Predicts bounding boxes and classes
    - Grid-based: 13x13 grid for 416x416 input
    - Anchors: 3 anchor boxes per grid cell
    
    Output format: [batch, grid_h, grid_w, num_anchors, (5 + num_classes)]
    where 5 = (x, y, w, h, objectness)
    
    Parameters: ~20M
    Input size: 416x416x3
    """
    
    def __init__(self, num_classes=20, num_anchors=3, input_size=416):
        """
        Initialize Custom Detector
        
        Args:
            num_classes: Number of object classes (default: 20)
            num_anchors: Number of anchor boxes per grid cell (default: 3)
            input_size: Input image size (default: 416)
        """
        super(CustomDetector, self).__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.input_size = input_size
        self.grid_size = input_size // 32  # After 5 max pools (2^5 = 32)
        
        # Backbone: Feature Extractor
        # Conv Block 1: 32 filters
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 416 -> 208
        )
        
        # Conv Block 2: 64 filters
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 208 -> 104
        )
        
        # Conv Block 3: 128 filters
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 104 -> 52
        )
        
        # Conv Block 4: 256 filters
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 52 -> 26
        )
        
        # Conv Block 5: 512 filters
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 26 -> 13
        )
        
        # Detection Head
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            # Output: num_anchors * (5 + num_classes)
            nn.Conv2d(1024, num_anchors * (5 + num_classes), kernel_size=1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Define default anchors (width, height) normalized to grid cell
        # These are typical anchor ratios for common objects
        self.anchors = torch.tensor([
            [1.3, 1.9],   # Anchor 1: Tall objects (person, bottle)
            [3.6, 2.8],   # Anchor 2: Wide objects (car, couch)
            [2.2, 2.2]    # Anchor 3: Square objects (chair, table)
        ])
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, 3, 416, 416)
            
        Returns:
            Predictions (batch_size, grid_h, grid_w, num_anchors, 5 + num_classes)
        """
        batch_size = x.size(0)
        
        # Backbone feature extraction
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        
        # Detection head
        x = self.detection_head(x)
        
        # Reshape output
        # From: (batch, num_anchors * (5 + num_classes), grid_h, grid_w)
        # To: (batch, grid_h, grid_w, num_anchors, 5 + num_classes)
        grid_h, grid_w = x.size(2), x.size(3)
        x = x.view(batch_size, self.num_anchors, 5 + self.num_classes, grid_h, grid_w)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        
        return x
    
    def _initialize_weights(self):
        """
        Initialize model weights using He initialization for LeakyReLU
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
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
    print("Custom Single-Stage Object Detector")
    print("=" * 70)
    
    # Create model
    num_classes = 20
    model = CustomDetector(num_classes=num_classes, num_anchors=3)
    
    # Model summary
    print(f"\n✓ Model created successfully")
    print(f"✓ Number of classes: {model.num_classes}")
    print(f"✓ Number of anchors: {model.num_anchors}")
    print(f"✓ Grid size: {model.grid_size}x{model.grid_size}")
    print(f"✓ Total parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    print(f"\n{'Testing forward pass...'}")
    dummy_input = torch.randn(2, 3, 416, 416)  # Batch of 2 images
    output = model(dummy_input)
    print(f"✓ Input shape: {dummy_input.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"  Expected: (batch, grid_h, grid_w, num_anchors, 5 + num_classes)")
    print(f"  Got: ({output.size(0)}, {output.size(1)}, {output.size(2)}, {output.size(3)}, {output.size(4)})")
    
    # Verify output dimensions
    expected_shape = (2, 13, 13, 3, 5 + num_classes)
    assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
    print(f"✓ Output shape verified!")
    
    # Architecture breakdown
    print(f"\n{'Architecture Breakdown:'}")
    print(f"  Backbone:")
    print(f"    Conv Block 1: 3 → 32 (MaxPool)")
    print(f"    Conv Block 2: 32 → 64 (MaxPool)")
    print(f"    Conv Block 3: 64 → 128 → 128 (MaxPool)")
    print(f"    Conv Block 4: 128 → 256 → 256 (MaxPool)")
    print(f"    Conv Block 5: 256 → 512 → 512 (MaxPool)")
    print(f"  Detection Head:")
    print(f"    512 → 1024 → 1024 → {3 * (5 + num_classes)}")
    print(f"  Output: {model.grid_size}×{model.grid_size} grid, {model.num_anchors} anchors/cell")
    
    print("\n" + "=" * 70)
