"""
Scene Classification Module using Custom ResNet-18
Classifies environment type (indoor/outdoor, scene category)
"""

import torch
import torch.nn as nn
from torchvision import models
import json
from pathlib import Path


class SceneClassifier:
    """
    Custom ResNet-18 based scene classification system
    
    Academic Justification:
    - ResNet-18: Efficient residual learning architecture
    - Custom Trained: Specifically optimized for 10 target scene classes
    - Skip connections prevent vanishing gradient
    - High performance in specialized scene recognition tasks
    """
    
    def __init__(self, model_path=None, use_gpu=True):
        """
        Initialize scene classifier
        
        Args:
            model_path: Path to fine-tuned model (None for default custom model)
            use_gpu: Use GPU if available
        """
        self.device = torch.device(
            'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        )
        print(f"✓ Using device: {self.device}")
        
        # Load Custom ResNet-18
        # Resolve path relative to this file to handle different working directories
        project_root = Path(__file__).resolve().parent.parent.parent
        local_model_path = project_root / "models" / "scene_classifier" / "custom_resnet18.pth"
        
        if model_path and Path(model_path).exists():
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)
            print(f"✓ Loaded custom model from {model_path}")
        elif local_model_path.exists():
            self.model = torch.load(local_model_path, map_location=self.device, weights_only=False)
            print(f"✓ Loaded CUSTOM TRAINED model from {local_model_path}")
        else:
            raise FileNotFoundError(
                f"CRITICAL ERROR: Custom model not found at {local_model_path}. "
                "Strictly custom mode requires a trained model. "
                "Please run train_model.py first."
            )
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load Custom Class Labels
        self.class_labels = self._load_custom_labels()
        
        # Define scene categories mapping
        self.scene_categories = self._define_scene_categories()
    
    def _load_custom_labels(self):
        """
        Load custom class labels
        
        Returns:
            dict: Class ID to label mapping
        """
        # Load from classes.txt if exists
        project_root = Path(__file__).resolve().parent.parent.parent
        classes_file = project_root / "models" / "scene_classifier" / "classes.txt"
        
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            return {i: name for i, name in enumerate(classes)}
            
        # Fallback to hardcoded list matching training script
        classes = [
            'beach', 'classroom', 'forest', 'gym', 'kitchen',
            'living_room', 'mountain', 'office', 'park', 'street'
        ]
        return {i: name for i, name in enumerate(classes)}
    
    def _define_scene_categories(self):
        """
        Define semantic scene categories
        
        Returns:
            dict: Scene category mappings
        """
        categories = {
            'indoor': [
                'office', 'kitchen', 'living_room', 'bedroom', 'bathroom',
                'restaurant', 'shop', 'classroom', 'library', 'gym'
            ],
            'outdoor': [
                'street', 'forest', 'beach', 'mountain', 'park'
            ],
            'commercial': [
                'restaurant', 'shop', 'office'
            ],
            'residential': [
                'kitchen', 'living_room', 'bedroom', 'bathroom'
            ],
            'educational': [
                'classroom', 'library'
            ],
            'natural': [
                'forest', 'beach', 'mountain', 'park'
            ],
            'urban': [
                'street', 'shop', 'restaurant'
            ]
        }
        
        return categories
    
    def classify_scene(self, image_tensor):
        """
        Classify scene from preprocessed image tensor
        
        Args:
            image_tensor: Preprocessed image tensor (1, 3, 224, 224)
            
        Returns:
            dict: Classification results with scene type and confidence
        """
        with torch.no_grad():
            # Move to device
            image_tensor = image_tensor.to(self.device)
            
            # Forward pass
            outputs = self.model(image_tensor)
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top predictions
            top_prob, top_idx = torch.topk(probabilities, min(5, len(self.class_labels)))
            
            # Convert to CPU and numpy
            top_prob = top_prob.cpu().numpy()[0]
            top_idx = top_idx.cpu().numpy()[0]
            
            # Map predictions to scene categories
            predictions = []
            for i in range(len(top_idx)):
                class_id = int(top_idx[i])
                class_name = self.class_labels.get(class_id, f"unknown_{class_id}")
                
                pred = {
                    'rank': i + 1,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': float(top_prob[i]),
                }
                predictions.append(pred)
        
        # Determine primary scene category
        primary_scene = predictions[0]['class_name']
        
        results = {
            'top_predictions': predictions,
            'primary_scene': primary_scene,
            'scene_type': self._get_scene_type(primary_scene),
            'confidence': predictions[0]['confidence']
        }
        
        return results
    
    def _get_scene_type(self, scene_category):
        """
        Get scene type (indoor/outdoor) from category
        
        Args:
            scene_category: Scene category string
            
        Returns:
            str: 'indoor' or 'outdoor'
        """
        indoor_scenes = [
            'office', 'kitchen', 'living_room', 'bedroom', 
            'bathroom', 'restaurant', 'shop', 'classroom', 'library', 'gym'
        ]
        
        if scene_category in indoor_scenes:
            return 'indoor'
        elif scene_category in self.scene_categories.get('outdoor', []):
            return 'outdoor'
        else:
            return 'unknown'
    
    def get_scene_attributes(self, scene_type):
        """
        Get attributes and characteristics of scene type
        
        Args:
            scene_type: Scene category
            
        Returns:
            dict: Scene attributes
        """
        attributes = {
            'street': {
                'typical_objects': ['vehicles', 'traffic lights', 'buildings', 'people'],
                'lighting': 'natural/artificial',
                'risk_level': 'medium',
                'typical_activities': ['transportation', 'commuting', 'walking']
            },
            'office': {
                'typical_objects': ['desk', 'computer', 'chair', 'documents'],
                'lighting': 'artificial',
                'risk_level': 'low',
                'typical_activities': ['working', 'meetings', 'computing']
            },
            'kitchen': {
                'typical_objects': ['appliances', 'utensils', 'food', 'sink'],
                'lighting': 'bright artificial',
                'risk_level': 'medium',
                'typical_activities': ['cooking', 'food preparation', 'cleaning']
            },
            'beach': {
                'typical_objects': ['sand', 'water', 'umbrellas', 'people'],
                'lighting': 'bright natural',
                'risk_level': 'low',
                'typical_activities': ['swimming', 'sunbathing', 'relaxing']
            },
            'forest': {
                'typical_objects': ['trees', 'plants', 'wildlife', 'trails'],
                'lighting': 'filtered natural',
                'risk_level': 'medium',
                'typical_activities': ['hiking', 'exploring', 'photography']
            }
        }
        
        return attributes.get(scene_type, {
            'typical_objects': [],
            'lighting': 'variable',
            'risk_level': 'unknown',
            'typical_activities': []
        })


# Example usage
if __name__ == "__main__":
    print("Scene Classification Module - Custom ResNet-18")
    print("=" * 60)
    print("Academic Justification:")
    print("• ResNet-18: Efficient 18-layer residual network")
    print("• Custom Trained: Optimized for 10 specific environment classes")
    print("• Strictly Custom: No pre-trained fallbacks used for classification")
    print("=" * 60)
    
    try:
        classifier = SceneClassifier()
        print(f"\n✓ Model loaded successfully")
        print(f"✓ Device: {classifier.device}")
    except Exception as e:
        print(f"\n❌ Error: {e}")