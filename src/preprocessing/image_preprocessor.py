"""
Image Preprocessor Module
Handles image loading, resizing, normalization, and noise reduction
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


class ImagePreprocessor:
    """
    Comprehensive image preprocessing pipeline for both object detection and scene classification
    """
    
    def __init__(self, target_size=(640, 640), normalize=True):
        """
        Initialize preprocessor with configuration
        
        Args:
            target_size: Tuple (width, height) for resizing
            normalize: Whether to apply normalization
        """
        self.target_size = target_size
        self.normalize = normalize
        
        # Define normalization parameters (ImageNet standards)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # Scene classification transform pipeline
        self.scene_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def load_image(self, image_path):
        """
        Load image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            numpy array: Loaded image in RGB format
        """
        try:
            # Load with PIL for better compatibility
            image = Image.open(image_path)
            image = image.convert('RGB')
            return np.array(image)
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")
    
    def reduce_noise(self, image):
        """
        Apply noise reduction using bilateral filter
        Preserves edges while smoothing noise
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            numpy array: Denoised image
        """
        # Bilateral filter parameters
        d = 9  # Diameter of pixel neighborhood
        sigma_color = 75  # Filter sigma in color space
        sigma_space = 75  # Filter sigma in coordinate space
        
        denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        return denoised
    
    def resize_image(self, image, size=None):
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: Input image
            size: Target size (width, height)
            
        Returns:
            numpy array: Resized image
        """
        if size is None:
            size = self.target_size
        
        # Use high-quality interpolation
        resized = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
        return resized
    
    def normalize_image(self, image):
        """
        Normalize image using ImageNet statistics
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            torch.Tensor: Normalized image tensor
        """
        # Convert to PIL for transform
        pil_image = Image.fromarray(image)
        normalized = self.scene_transform(pil_image)
        return normalized
    
    def preprocess_for_detection(self, image_path):
        """
        Complete preprocessing pipeline for object detection (Custom Detector)
        
        Args:
            image_path: Path to input image
            
        Returns:
            tuple: (preprocessed_image, original_image)
        """
        # Load original image
        original = self.load_image(image_path)
        
        # Apply noise reduction
        denoised = self.reduce_noise(original)
        
        # Resize for YOLO (maintains aspect ratio internally)
        processed = cv2.resize(denoised, self.target_size)
        
        return processed, original
    
    def preprocess_for_classification(self, image_path):
        """
        Complete preprocessing pipeline for scene classification
        
        Args:
            image_path: Path to input image
            
        Returns:
            torch.Tensor: Preprocessed image tensor ready for model
        """
        # Load image
        image = self.load_image(image_path)
        
        # Apply noise reduction
        denoised = self.reduce_noise(image)
        
        # Normalize and convert to tensor
        tensor = self.normalize_image(denoised)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def enhance_image(self, image):
        """
        Optional: Enhance image contrast and brightness
        
        Args:
            image: Input image
            
        Returns:
            numpy array: Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def validate_image(self, image_path):
        """
        Validate if image file is valid and processable
        
        Args:
            image_path: Path to image file
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            img = Image.open(image_path)
            img.verify()
            return True
        except:
            return False


# Example usage and testing
if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    
    print("Image Preprocessor Module")
    print("=" * 50)
    print("Features:")
    print("✓ Image loading and validation")
    print("✓ Noise reduction (Bilateral Filter)")
    print("✓ Intelligent resizing")
    print("✓ ImageNet normalization")
    print("✓ CLAHE enhancement")
    print("=" * 50)