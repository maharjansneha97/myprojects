"""
Updated Scene Classifier - Uses Custom Trained Model
Replaces pretrained CLIP with custom ResNet-18
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.custom_scene_classifier import CustomSceneClassifier as CustomModel


class SceneClassifier:
    """
    Scene Classifier using Custom Trained ResNet-18
    NO PRETRAINED WEIGHTS - Fully custom trained model
    """
    
    def __init__(self, model_path='models/scene_classifier_aid/best_model.pth', use_gpu=True):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"✓ Custom Scene Classifier using device: {self.device}")
        
        # Resolve absolute path to model
        # If path is relative, make it relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        model_path_obj = Path(model_path)
        
        if not model_path_obj.is_absolute():
            checkpoint_file = project_root / model_path
        else:
            checkpoint_file = model_path_obj
            
        if not checkpoint_file.exists():
            print(f"DEBUG: Project Root: {project_root}")
            print(f"DEBUG: Looking for model at: {checkpoint_file}")
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_file}")
        
        print(f"Loading custom model from: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        
        # Get model info
        self.num_classes = checkpoint['num_classes']
        self.classes = checkpoint['classes']
        
        # Create and load model
        self.model = CustomModel(num_classes=self.num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Custom ResNet-18 loaded successfully")
        print(f"  • Number of classes: {self.num_classes}")
        print(f"  • Training accuracy: {checkpoint.get('val_acc', 'N/A')}")
        print(f"  • NO PRETRAINED WEIGHTS - Fully custom trained")
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Scene categories for compatibility
        self.scene_categories = {
            'indoor': ['office', 'kitchen', 'living_room', 'bedroom', 'bathroom', 'restaurant', 'shop', 'classroom', 'gym'],
            'outdoor': ['street', 'forest', 'beach', 'mountain', 'park', 'city']
        }
    
    def classify_scene(self, image_input):
        """
        Classify scene using custom trained model
        
        Args:
            image_input: Can be file path (str) or PIL Image
        
        Returns:
            dict: Classification results with top predictions
        """
        # Load image
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
        else:
            raise ValueError("Expected file path or PIL Image")
        
        # Preprocess
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            
            # Get top 5 predictions
            top_probs, top_indices = torch.topk(probs, min(5, self.num_classes), dim=1)
        
        # Format results
        predictions = []
        for i in range(top_indices.shape[1]):
            idx = top_indices[0][i].item()
            class_name = self.classes[idx]
            confidence = top_probs[0][i].item()
            
            predictions.append({
                'rank': i + 1,
                'class_name': class_name,
                'confidence': confidence
            })
        
        primary_scene = predictions[0]['class_name']
        
        return {
            'top_predictions': predictions,
            'primary_scene': primary_scene,
            'scene_type': self._get_scene_type(primary_scene),
            'confidence': predictions[0]['confidence']
        }
    
    def _get_scene_type(self, scene_name):
        """Determine if scene is indoor or outdoor"""
        scene_lower = scene_name.lower()
        if scene_lower in self.scene_categories['indoor']:
            return 'indoor'
        elif scene_lower in self.scene_categories['outdoor']:
            return 'outdoor'
        return 'unknown'


# Alias for compatibility
HFSceneClassifier = SceneClassifier


if __name__ == "__main__":
    # Test the classifier
    print("\n" + "="*60)
    print("TESTING CUSTOM SCENE CLASSIFIER")
    print("="*60)
    
    try:
        classifier = SceneClassifier()
        print("\n✓ Model loaded successfully!")
        print(f"  Classes: {classifier.classes}")
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("  Please train the model first using:")
        print("  python src/training/train_scene_classifier.py")
