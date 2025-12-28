"""
Check Inference for Custom Models
Loads the trained models and runs inference on a sample image.
"""
import torch
import cv2
import numpy as np
from pathlib import Path
import sys

# Add project root to path
# Script is in src/utils/
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Check if src is importable
try:
    from src.detection.object_detector import ObjectDetector
    from src.classification.scene_classifier import SceneClassifier
except ImportError:
    # Try adding parent of src if not found
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.detection.object_detector import ObjectDetector
    from src.classification.scene_classifier import SceneClassifier

def check_detector():
    print("\n--- Checking Object Detector ---")
    try:
        detector = ObjectDetector()
        print("✓ Detector loaded successfully")
        
        # improved test: finding a real image from dataset
        dataset_dir = project_root / "dataset" / "detection" / "images"
        images = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.jpeg"))
        
        if not images:
            print("! No images found in dataset/detection/images to test.")
            return

        test_img = images[0]
        print(f"Testing on image: {test_img}")
        
        results = detector.detect_objects(str(test_img), save_output=True, output_path="test_detection_output.jpg")
        print(f"✓ Inference successful. Detected {len(results['detections'])} objects.")
        for det in results['detections']:
            print(f"  - {det['class_name']} ({det['confidence']:.2f})")
            
        print("✓ Output saved to test_detection_output.jpg")
        
    except Exception as e:
        print(f"❌ Detector Verification Failed: {e}")
        import traceback
        traceback.print_exc()

def check_classifier():
    print("\n--- Checking Scene Classifier ---")
    try:
        classifier = SceneClassifier()
        print("✓ Classifier loaded successfully")
        
        # Wait for training to finish?? 
        # We can check if model exists, if not it will load random weights or fail (handled in init)
        
        # Test on a scene image
        scene_dir = project_root / "dataset" / "scenes" / "val"
        # Find any subdirectory
        subdirs = [d for d in scene_dir.iterdir() if d.is_dir()]
        if not subdirs:
             # Try flat structure if val not populated yet
             scene_dir = project_root / "dataset" / "scenes"
             subdirs = [d for d in scene_dir.iterdir() if d.is_dir() and d.name not in ['train', 'val', 'test']]
        
        if subdirs:
            class_dir = subdirs[0]
            images = list(class_dir.glob("*.jpg"))
            if images:
                test_img = images[0]
                print(f"Testing on image: {test_img}")
                
                # Need preprocessing matching training
                img = cv2.imread(str(test_img))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img.transpose(2, 0, 1)
                img = img / 255.0
                img = (img - np.array([0.485, 0.456, 0.406]).reshape(3,1,1)) / np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
                img_tensor = torch.from_numpy(img).float().unsqueeze(0)
                
                results = classifier.classify_scene(img_tensor)
                print(f"✓ Classification successful.")
                print(f"  - Scene: {results['primary_scene']} ({results['confidence']:.2f})")
                print(f"  - Type: {results['scene_type']}")
            else:
                print("! No images found in first scene directory.")
        else:
             print("! No scene directories found.")
             
    except Exception as e:
        print(f"❌ Classifier Verification Failed: {e}")

if __name__ == "__main__":
    check_detector()
    check_classifier()
