"""
Updated Object Detector - Uses Custom Trained Model
Replaces pretrained DETR with custom Faster R-CNN
"""

import torch
from PIL import Image, ImageDraw
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.custom_object_detector import CustomObjectDetector as CustomModel


class ObjectDetector:
    """
    Object Detector using Custom Trained Faster R-CNN
    NO PRETRAINED WEIGHTS - Fully custom trained model
    """
    
    def __init__(self, model_path='models/object_detector/best_model.pth', 
                 confidence_threshold=0.3, use_gpu=True):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        print(f"✓ Custom Object Detector using device: {self.device}")
        
        # Resolve absolute path to model
        # If path is relative, make it relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        model_path_obj = Path(model_path)
        
        if not model_path_obj.is_absolute():
            checkpoint_file = project_root / model_path
        else:
            checkpoint_file = model_path_obj
            
        if not checkpoint_file.exists():
            # Try looking one level up just in case (e.g. if running from src/)
            if (Path('..') / model_path).exists():
                checkpoint_file = Path('..') / model_path
            else:
                print(f"DEBUG: Project Root: {project_root}")
                print(f"DEBUG: Looking for model at: {checkpoint_file}")
                raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_file}")
        
        print(f"Loading custom model from: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        
        # Get model info
        self.num_classes = checkpoint['num_classes']
        
        # Define VOC categories (1-indexed for Model Output alignment)
        # Faster R-CNN predicts 1-20 for VOC, 0 is background.
        voc_labels = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        # Map Model Output Label -> Class Name
        # e.g., Model Label 15 (Person) -> 'person'
        self.categories = {i+1: name for i, name in enumerate(voc_labels)}
        
        # Create and load model
        self.model = CustomModel(num_classes=self.num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Custom Faster R-CNN loaded successfully")
        print(f"  • Number of classes: {self.num_classes}")
        print(f"  • NO PRETRAINED WEIGHTS - Fully custom trained")
    
    def detect_objects(self, image_path, save_output=False, output_path=None, use_tiling=True):
        """
        Detect objects in an image using custom trained model
        With Optional Tiling Inference for small objects
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            width, height = image.size
            
            # Convert to tensor
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.Resize((800, 800)),
                transforms.ToTensor()
            ])
            
            all_boxes = []
            all_labels = []
            all_scores = []
            
            # 1. Full Image Inference
            img_tensor = transform(image).to(self.device)
            predictions = self.model.predict([img_tensor], confidence_threshold=self.confidence_threshold)
            pred = predictions[0]
            
            # Scale boxes from 800x800 back to original size
            x_scale = width / 800.0
            y_scale = height / 800.0
            
            for i in range(len(pred['boxes'])):
                box = pred['boxes'][i].cpu().numpy()
                all_boxes.append([box[0] * x_scale, box[1] * y_scale, box[2] * x_scale, box[3] * y_scale])
                all_labels.append(pred['labels'][i].item())
                all_scores.append(pred['scores'][i].item())
            
            # 2. Tiling Inference (4 Overlapping Quadrants)
            if use_tiling:
                # Define 4 quadrants with 15% overlap
                overlap = 0.15
                mid_x, mid_y = width // 2, height // 2
                offset_x, offset_y = int(mid_x * overlap), int(mid_y * overlap)
                
                tiles = [
                    (0, 0, mid_x + offset_x, mid_y + offset_y), # Top-Left
                    (mid_x - offset_x, 0, width, mid_y + offset_y), # Top-Right
                    (0, mid_y - offset_y, mid_x + offset_x, height), # Bottom-Left
                    (mid_x - offset_x, mid_y - offset_y, width, height) # Bottom-Right
                ]
                
                for tx1, ty1, tx2, ty2 in tiles:
                    tile_img = image.crop((tx1, ty1, tx2, ty2))
                    tw, th = tx2 - tx1, ty2 - ty1
                    
                    tile_tensor = transform(tile_img).to(self.device)
                    # Use higher threshold for tiles to reduce noise
                    tile_preds = self.model.predict([tile_tensor], confidence_threshold=self.confidence_threshold + 0.1)
                    t_pred = tile_preds[0]
                    
                    tx_scale = tw / 800.0
                    ty_scale = th / 800.0
                    
                    for i in range(len(t_pred['boxes'])):
                        box = t_pred['boxes'][i].cpu().numpy()
                        # Map tile box back to original image coordinates
                        all_boxes.append([
                            box[0] * tx_scale + tx1, 
                            box[1] * ty_scale + ty1, 
                            box[2] * tx_scale + tx1, 
                            box[3] * ty_scale + ty1
                        ])
                        all_labels.append(t_pred['labels'][i].item())
                        all_scores.append(t_pred['scores'][i].item())

            # 3. Merge Results using Non-Maximum Suppression (NMS)
            from torchvision.ops import nms
            if all_boxes:
                boxes_tensor = torch.as_tensor(all_boxes, dtype=torch.float32)
                scores_tensor = torch.as_tensor(all_scores, dtype=torch.float32)
                # Keep indices after NMS (0.45 IoU threshold)
                keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.45)
                
                final_boxes = boxes_tensor[keep].numpy()
                final_labels = [all_labels[i] for i in keep]
                final_scores = [all_scores[i] for i in keep]
            else:
                final_boxes, final_labels, final_scores = [], [], []

            # 4. Final Polish: Confidence Boosting & Background Suppression
            detections = []
            
            # Define category colors for better visual presentation
            COLORS = {
                'person': 'lime',
                'vehicle': 'cyan', # car, bus, train, motorbike, bicycle
                'animal': 'yellow', # dog, cat, horse, sheep, cow, bird
                'indoor': 'magenta', # sofa, chair, tvmonitor, diningtable, bottle, pottedplant
                'default': 'red'
            }
            category_map = {
                'car': 'vehicle', 'bus': 'vehicle', 'train': 'vehicle', 'motorbike': 'vehicle', 'bicycle': 'vehicle',
                'dog': 'animal', 'cat': 'animal', 'horse': 'animal', 'sheep': 'animal', 'cow': 'animal', 'bird': 'animal',
                'sofa': 'indoor', 'chair': 'indoor', 'tvmonitor': 'indoor', 'diningtable': 'indoor', 'bottle': 'indoor', 'pottedplant': 'indoor'
            }

            if save_output:
                draw = ImageDraw.Draw(image)
                # Dynamic font scaling based on image size
                font_size = max(20, int(width / 40))
                try:
                    from PIL import ImageFont
                    # Try to load a bold font
                    font = ImageFont.truetype("arialbd.ttf", font_size)
                except:
                    font = None # Fallback to default
            
            for i in range(len(final_boxes)):
                box = final_boxes[i]
                label_id = final_labels[i]
                score = float(final_scores[i])
                class_name = self.categories.get(label_id, f'Unknown_{label_id}')
                
                # A. Background Suppression: If box is > 70% of image and low confidence, skip it
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                img_area = width * height
                if box_area > (0.7 * img_area) and score < 0.4:
                    continue

                # B. Semantic Confidence Boosting: (Optional/Heuristic)
                # Small boost for very common objects to reduce flickering
                if class_name in ['person', 'car'] and score < 0.5:
                    score = min(0.99, score + 0.05)

                detections.append({
                    'class_name': class_name,
                    'confidence': round(score, 2),
                    'box': [round(float(x), 2) for x in box]
                })
                
                if save_output:
                    category = category_map.get(class_name, 'person' if class_name == 'person' else 'default')
                    color = COLORS.get(category, 'red')
                    
                    # Draw bold bounding box (multiple rectangles for thickness)
                    for offset in range(4):
                        draw.rectangle([box[0]-offset, box[1]-offset, box[2]+offset, box[3]+offset], outline=color)
                    
                    # Draw label background for better readability
                    label_text = f"{class_name.upper()} {score:.2f}"
                    text_bbox = draw.textbbox((box[0], box[1]), label_text, font=font)
                    draw.rectangle(text_bbox, fill=color)
                    draw.text((box[0], box[1]), label_text, fill="black", font=font)
            
            if save_output and output_path:
                image.save(output_path)
            
            return {
                'detections': detections,
                'detection_count': len(detections),
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Error in Custom Object Detection: {e}")
            return {
                'detections': [],
                'detection_count': 0,
                'status': 'error',
                'error': str(e)
            }


# Alias for compatibility
HFObjectDetector = ObjectDetector


if __name__ == "__main__":
    # Test the detector
    print("\n" + "="*60)
    print("TESTING CUSTOM OBJECT DETECTOR")
    print("="*60)
    
    try:
        detector = ObjectDetector()
        print("\n✓ Model loaded successfully!")
        print(f"  Categories: {detector.categories}")
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("  Please train the model first using:")
        print("  python src/training/train_object_detector.py")
