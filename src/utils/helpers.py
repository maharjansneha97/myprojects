"""
Helper Utilities Module
Common utility functions for file handling, visualization, and logging
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Any


class FileHandler:
    """Handle file operations"""
    
    @staticmethod
    def ensure_directory(directory_path):
        """Create directory if it doesn't exist"""
        Path(directory_path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def save_json(data, filepath):
        """Save data as JSON"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    
    @staticmethod
    def load_json(filepath):
        """Load JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def get_supported_images(directory):
        """Get all supported image files in directory"""
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in supported_formats:
            image_files.extend(Path(directory).glob(f'*{ext}'))
            image_files.extend(Path(directory).glob(f'*{ext.upper()}'))
        
        return sorted(image_files)


class Visualizer:
    """Visualization utilities"""
    
    @staticmethod
    def draw_detections(image, detections, thickness=2):
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: Input image (numpy array)
            detections: List of detection dictionaries
            thickness: Line thickness
            
        Returns:
            numpy array: Annotated image
        """
        annotated = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Extract coordinates
            x1, y1, x2, y2 = map(int, bbox)
            
            # Random color based on class
            color = Visualizer._get_color_for_class(class_name)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Create label
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                annotated, 
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return annotated
    
    @staticmethod
    def _get_color_for_class(class_name):
        """Generate consistent color for class"""
        # Simple hash-based color generation
        hash_val = sum(ord(c) for c in class_name)
        
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128)
        ]
        
        return colors[hash_val % len(colors)]
    
    @staticmethod
    def create_result_grid(images, titles, save_path=None):
        """
        Create grid of result images
        
        Args:
            images: List of images
            titles: List of titles
            save_path: Path to save grid
        """
        n = len(images)
        cols = 2
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
        axes = axes.flatten() if n > 1 else [axes]
        
        for i, (img, title) in enumerate(zip(images, titles)):
            if img.dtype == np.uint8 and len(img.shape) == 3:
                # Convert BGR to RGB if needed
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(img)
            axes[i].set_title(title, fontsize=12, fontweight='bold')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(images), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Grid saved to {save_path}")
        
        plt.close()


class Logger:
    """Simple logging utility"""
    
    def __init__(self, log_file='logs/system.log'):
        self.log_file = log_file
        FileHandler.ensure_directory(os.path.dirname(log_file))
    
    def log(self, message, level='INFO'):
        """Write log message"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        print(log_entry.strip())
    
    def info(self, message):
        self.log(message, 'INFO')
    
    def warning(self, message):
        self.log(message, 'WARNING')
    
    def error(self, message):
        self.log(message, 'ERROR')


class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {}
    
    def start(self):
        """Start timing"""
        self.start_time = datetime.now()
    
    def stop(self, operation_name):
        """Stop timing and record"""
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            self.metrics[operation_name] = elapsed
            return elapsed
        return 0
    
    def get_metrics(self):
        """Get all recorded metrics"""
        return self.metrics
    
    def print_metrics(self):
        """Print performance metrics"""
        print("\n" + "=" * 60)
        print("PERFORMANCE METRICS")
        print("=" * 60)
        
        for operation, time_taken in self.metrics.items():
            print(f"{operation:.<40} {time_taken:.3f}s")
        
        if self.metrics:
            total = sum(self.metrics.values())
            print("-" * 60)
            print(f"{'Total Time':.<40} {total:.3f}s")
        
        print("=" * 60)


class ConfigManager:
    """Manage configuration settings"""
    
    DEFAULT_CONFIG = {
        'object_detection': {
            'model': 'yolov8n.pt',
            'confidence_threshold': 0.25,
            'iou_threshold': 0.45
        },
        'scene_classification': {
            'model': 'resnet50',
            'use_gpu': True
        },
        'preprocessing': {
            'target_size': [640, 640],
            'apply_denoising': True,
            'apply_enhancement': False
        },
        'interpretation': {
            'detailed_mode': True,
            'risk_assessment': True
        },
        'output': {
            'save_annotated_images': True,
            'save_json_results': True,
            'output_directory': 'results/output_images'
        }
    }
    
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file or use defaults"""
        if os.path.exists(self.config_path):
            return FileHandler.load_json(self.config_path)
        else:
            self.save_config(self.DEFAULT_CONFIG)
            return self.DEFAULT_CONFIG
    
    def save_config(self, config):
        """Save configuration to file"""
        FileHandler.save_json(config, self.config_path)
    
    def get(self, key, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key, value):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self.save_config(self.config)


# Example usage
if __name__ == "__main__":
    print("Helper Utilities Module")
    print("=" * 60)
    print("Available utilities:")
    print("• FileHandler: File and directory operations")
    print("• Visualizer: Image annotation and visualization")
    print("• Logger: System logging")
    print("• PerformanceMonitor: Timing and metrics")
    print("• ConfigManager: Configuration management")
    print("=" * 60)