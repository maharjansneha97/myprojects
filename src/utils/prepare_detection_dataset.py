"""
Dataset Auto-Labeling Script for Object Detection
Uses pre-trained YOLOv8 model to generate labels for the collected images.
Maps COCO classes (80) to Pascal VOC classes (20).
"""

import os
from pathlib import Path
import argparse
from tqdm import tqdm
import cv2
from ultralytics import YOLO

# Pascal VOC 20 classes (Indices 0-19)
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# COCO Class ID to VOC Class ID Mapping
# COCO has 80 classes. We map relevant ones to VOC's 0-19.
# Returns None if COCO class is not in VOC.
def get_voc_id_from_coco(coco_id, coco_names):
    coco_name = coco_names[coco_id]
    
    # Direct matches
    if coco_name in VOC_CLASSES:
        return VOC_CLASSES.index(coco_name)
    
    # Name variations
    mapping = {
        'airplane': 'aeroplane',
        'motorcycle': 'motorbike',
        'potted plant': 'pottedplant',
        'dining table': 'diningtable',
        'couch': 'sofa',
        'tv': 'tvmonitor'
    }
    
    if coco_name in mapping:
        return VOC_CLASSES.index(mapping[coco_name])
        
    return None

def auto_label_images(base_dir):
    img_dir = Path(base_dir) / 'images'
    label_dir = Path(base_dir) / 'labels'
    label_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading YOLOv8n model for auto-labeling...")
    model = YOLO('yolov8n.pt') 
    
    # Get all image files
    image_files = list(img_dir.glob('*'))
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in image_files if f.suffix.lower() in valid_extensions]
    
    print(f"Found {len(image_files)} images. Starting auto-labeling...")
    
    count = 0
    
    for img_path in tqdm(image_files):
        # Run inference
        results = model(str(img_path), verbose=False, conf=0.25) # Confidence threshold 0.25
        
        result = results[0]
        coco_names = result.names
        
        labels = []
        
        for box in result.boxes:
            # box.cls is tensor, box.xywhn is tensor (normalized xywh)
            coco_id = int(box.cls[0].item())
            voc_id = get_voc_id_from_coco(coco_id, coco_names)
            
            if voc_id is not None:
                # Get normalized coordinates
                xywhn = box.xywhn[0].tolist() # x_center, y_center, width, height
                
                # Create label line: class_id x y w h
                label_line = f"{voc_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}"
                labels.append(label_line)
        
        # Save label file if we found valid objects, OR save empty file?
        # Standard YOLO training often allows background images (empty label files).
        # But for stable training of a custom detector, it's often better to have positive samples.
        # We will save the file regardless. If empty, it's a negative sample (no target objects).
        
        label_filename = img_path.stem + ".txt"
        save_path = label_dir / label_filename
        
        with open(save_path, 'w') as f:
            if labels:
                f.write('\n'.join(labels))
                
        count += 1
            
    print(f"\n✓ Auto-labeling complete.")
    print(f"Labels saved to: {label_dir}")

def main(args):
    print("=" * 70)
    print("Auto-Labeling Object Detection Dataset")
    print("=" * 70)
    
    try:
        auto_label_images(args.output_dir)
    except Exception as e:
        print(f"\n❌ Error during auto-labeling: {e}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Auto-label images using YOLOv8')
    parser.add_argument('--output-dir', type=str, default='dataset/detection',
                       help='Output directory containing images/')
    
    args = parser.parse_args()
    main(args)
