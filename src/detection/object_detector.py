"""
Object Detection Module using YOLOv8
Detects objects in environment images with bounding boxes and confidence scores
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path


class ObjectDetector:
    """
    YOLOv8-based object detection system
    Academic Justification:
    - State-of-the-art performance
    - Real-time inference capability
    - Pre-trained on COCO dataset (80 classes)
    - Widely accepted in computer vision research
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.25):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_path: Path to custom trained model (None for pretrained)
            confidence_threshold: Minimum confidence for detection (0-1)
        """
        self.confidence_threshold = confidence_threshold
        
        # Load model (downloads pretrained if not exists)
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            print(f"✓ Loaded custom model from {model_path}")
        else:
            # Check for local model in models/yolo
            local_model_path = Path("models/yolo/yolov8n.pt")
            if local_model_path.exists():
                self.model = YOLO(str(local_model_path))
                print(f"✓ Loaded local model from {local_model_path}")
            else:
                self.model = YOLO('yolov8n.pt')  # Nano model for speed
                print("✓ Loaded YOLOv8n pretrained model")
        
        # COCO dataset class names
        self.class_names = self.model.names
        
    def detect_objects(self, image_path, save_output=False, output_path=None):
        """
        Perform object detection on image
        
        Args:
            image_path: Path to input image
            save_output: Whether to save annotated image
            output_path: Path to save annotated image
            
        Returns:
            dict: Detection results containing:
                - detections: List of detected objects
                - annotated_image: Image with bounding boxes
                - detection_count: Number of objects detected
        """
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=self.confidence_threshold,
            save=False,
            verbose=False
        )
        
        # Extract results
        result = results[0]
        boxes = result.boxes
        
        # Parse detections
        detections = []
        for box in boxes:
            detection = {
                'class_id': int(box.cls[0]),
                'class_name': self.class_names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
            }
            detections.append(detection)
        
        # Get annotated image
        annotated_image = result.plot()
        
        # Save if requested
        if save_output and output_path:
            cv2.imwrite(output_path, annotated_image)
        
        # Prepare results
        results_dict = {
            'detections': detections,
            'annotated_image': annotated_image,
            'detection_count': len(detections),
            'image_shape': result.orig_shape
        }
        
        return results_dict
    
    def get_object_summary(self, detections):
        """
        Generate summary statistics of detected objects
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            dict: Summary with object counts and categories
        """
        object_counts = {}
        total_confidence = 0
        
        for det in detections:
            class_name = det['class_name']
            confidence = det['confidence']
            
            if class_name in object_counts:
                object_counts[class_name]['count'] += 1
                object_counts[class_name]['avg_confidence'] += confidence
            else:
                object_counts[class_name] = {
                    'count': 1,
                    'avg_confidence': confidence
                }
            
            total_confidence += confidence
        
        # Calculate average confidence per class
        for obj in object_counts:
            count = object_counts[obj]['count']
            object_counts[obj]['avg_confidence'] /= count
            object_counts[obj]['avg_confidence'] = round(
                object_counts[obj]['avg_confidence'], 3
            )
        
        summary = {
            'object_counts': object_counts,
            'total_objects': len(detections),
            'unique_objects': len(object_counts),
            'avg_confidence': round(
                total_confidence / len(detections), 3
            ) if detections else 0
        }
        
        return summary
    
    def filter_detections_by_confidence(self, detections, min_confidence=0.5):
        """
        Filter detections by confidence threshold
        
        Args:
            detections: List of detection dictionaries
            min_confidence: Minimum confidence threshold
            
        Returns:
            list: Filtered detections
        """
        return [det for det in detections if det['confidence'] >= min_confidence]
    
    def categorize_objects(self, detections):
        """
        Categorize detected objects into semantic groups
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            dict: Categorized objects
        """
        categories = {
            'people': [],
            'vehicles': [],
            'animals': [],
            'indoor_objects': [],
            'outdoor_objects': [],
            'electronics': [],
            'furniture': [],
            'other': []
        }
        
        # Define category mappings
        people_classes = ['person']
        vehicle_classes = ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
                          'train', 'truck', 'boat']
        animal_classes = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
                         'elephant', 'bear', 'zebra', 'giraffe']
        indoor_classes = ['chair', 'couch', 'bed', 'dining table', 'toilet', 
                         'tv', 'laptop', 'mouse', 'keyboard', 'book', 'clock']
        outdoor_classes = ['traffic light', 'fire hydrant', 'stop sign', 
                          'parking meter', 'bench']
        electronics_classes = ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 
                              'cell phone', 'microwave', 'oven', 'toaster']
        furniture_classes = ['chair', 'couch', 'bed', 'dining table']
        
        # Categorize each detection
        for det in detections:
            class_name = det['class_name']
            
            if class_name in people_classes:
                categories['people'].append(det)
            elif class_name in vehicle_classes:
                categories['vehicles'].append(det)
            elif class_name in animal_classes:
                categories['animals'].append(det)
            elif class_name in indoor_classes:
                categories['indoor_objects'].append(det)
            elif class_name in outdoor_classes:
                categories['outdoor_objects'].append(det)
            elif class_name in electronics_classes:
                categories['electronics'].append(det)
            elif class_name in furniture_classes:
                categories['furniture'].append(det)
            else:
                categories['other'].append(det)
        
        # Remove empty categories
        categories = {k: v for k, v in categories.items() if v}
        
        return categories


# Example usage
if __name__ == "__main__":
    print("Object Detection Module - YOLOv8")
    print("=" * 60)
    print("Academic Justification:")
    print("• YOLOv8: Latest version of You Only Look Once architecture")
    print("• Pre-trained on COCO dataset (80 object classes)")
    print("• Real-time performance suitable for practical applications")
    print("• High accuracy with efficient inference")
    print("=" * 60)
    
    detector = ObjectDetector()
    print(f"\n✓ Model loaded successfully")
    print(f"✓ Confidence threshold: {detector.confidence_threshold}")
    print(f"✓ Number of classes: {len(detector.class_names)}")