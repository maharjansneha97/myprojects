"""
Detection Utilities for Custom Object Detector
Helper functions for training and inference

Functions:
- IoU calculation
- Non-Maximum Suppression (NMS)
- Bounding box encoding/decoding
- Anchor box generation
- mAP calculation
"""

import torch
import numpy as np


def calculate_iou(box1, box2, box_format='xyxy'):
    """
    Calculate Intersection over Union (IoU) between two boxes
    
    Args:
        box1: First box [x1, y1, x2, y2] or [x_center, y_center, width, height]
        box2: Second box in same format
        box_format: 'xyxy' or 'xywh'
        
    Returns:
        float: IoU value between 0 and 1
    """
    if box_format == 'xywh':
        # Convert from center format to corner format
        box1 = xywh_to_xyxy(box1)
        box2 = xywh_to_xyxy(box2)
    
    # Get coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / (union_area + 1e-6)
    
    return iou


def batch_iou(boxes1, boxes2, box_format='xyxy'):
    """
    Calculate IoU between two sets of boxes
    
    Args:
        boxes1: Tensor of shape (N, 4)
        boxes2: Tensor of shape (M, 4)
        box_format: 'xyxy' or 'xywh'
        
    Returns:
        Tensor of shape (N, M) with IoU values
    """
    if box_format == 'xywh':
        boxes1 = xywh_to_xyxy_batch(boxes1)
        boxes2 = xywh_to_xyxy_batch(boxes2)
    
    # Expand dimensions for broadcasting
    boxes1 = boxes1.unsqueeze(1)  # (N, 1, 4)
    boxes2 = boxes2.unsqueeze(0)  # (1, M, 4)
    
    # Calculate intersection
    inter_x_min = torch.max(boxes1[..., 0], boxes2[..., 0])
    inter_y_min = torch.max(boxes1[..., 1], boxes2[..., 1])
    inter_x_max = torch.min(boxes1[..., 2], boxes2[..., 2])
    inter_y_max = torch.min(boxes1[..., 3], boxes2[..., 3])
    
    inter_width = torch.clamp(inter_x_max - inter_x_min, min=0)
    inter_height = torch.clamp(inter_y_max - inter_y_min, min=0)
    inter_area = inter_width * inter_height
    
    # Calculate union
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union_area = boxes1_area + boxes2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / (union_area + 1e-6)
    
    return iou


def non_maximum_suppression(predictions, conf_threshold=0.5, iou_threshold=0.4):
    """
    Apply Non-Maximum Suppression to remove duplicate detections
    
    Args:
        predictions: List of detections [x1, y1, x2, y2, confidence, class_id]
        conf_threshold: Minimum confidence threshold
        iou_threshold: IoU threshold for NMS
        
    Returns:
        List of filtered detections
    """
    if len(predictions) == 0:
        return []
    
    # Convert to numpy for easier manipulation
    predictions = np.array(predictions)
    
    # Filter by confidence
    mask = predictions[:, 4] >= conf_threshold
    predictions = predictions[mask]
    
    if len(predictions) == 0:
        return []
    
    # Sort by confidence (descending)
    indices = np.argsort(predictions[:, 4])[::-1]
    predictions = predictions[indices]
    
    keep = []
    
    while len(predictions) > 0:
        # Keep the detection with highest confidence
        keep.append(predictions[0])
        
        if len(predictions) == 1:
            break
        
        # Calculate IoU with remaining boxes
        ious = []
        for i in range(1, len(predictions)):
            iou = calculate_iou(predictions[0][:4], predictions[i][:4], box_format='xyxy')
            ious.append(iou)
        
        ious = np.array(ious)
        
        # Remove boxes with high IoU
        mask = ious < iou_threshold
        predictions = predictions[1:][mask]
    
    return keep


def xywh_to_xyxy(box):
    """
    Convert box from (x_center, y_center, width, height) to (x1, y1, x2, y2)
    
    Args:
        box: [x_center, y_center, width, height]
        
    Returns:
        [x1, y1, x2, y2]
    """
    x_center, y_center, width, height = box
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return [x1, y1, x2, y2]


def xyxy_to_xywh(box):
    """
    Convert box from (x1, y1, x2, y2) to (x_center, y_center, width, height)
    
    Args:
        box: [x1, y1, x2, y2]
        
    Returns:
        [x_center, y_center, width, height]
    """
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return [x_center, y_center, width, height]


def xywh_to_xyxy_batch(boxes):
    """
    Convert boxes from xywh to xyxy format (batch version)
    
    Args:
        boxes: Tensor of shape (N, 4) in xywh format
        
    Returns:
        Tensor of shape (N, 4) in xyxy format
    """
    x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def xyxy_to_xywh_batch(boxes):
    """
    Convert boxes from xyxy to xywh format (batch version)
    
    Args:
        boxes: Tensor of shape (N, 4) in xyxy format
        
    Returns:
        Tensor of shape (N, 4) in xywh format
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return torch.stack([x_center, y_center, width, height], dim=1)


def generate_anchors(grid_size, num_anchors=3, input_size=416):
    """
    Generate anchor boxes for all grid cells
    
    Args:
        grid_size: Size of the grid (e.g., 13 for 13x13)
        num_anchors: Number of anchors per grid cell
        input_size: Input image size
        
    Returns:
        Tensor of shape (grid_size, grid_size, num_anchors, 4)
    """
    # Default anchor sizes (width, height) relative to grid cell
    anchor_sizes = torch.tensor([
        [1.3, 1.9],   # Tall objects
        [3.6, 2.8],   # Wide objects
        [2.2, 2.2]    # Square objects
    ])
    
    anchors = torch.zeros(grid_size, grid_size, num_anchors, 4)
    
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(num_anchors):
                # Center of grid cell (normalized to 0-1)
                cx = (j + 0.5) / grid_size
                cy = (i + 0.5) / grid_size
                
                # Anchor size (normalized to 0-1)
                w = anchor_sizes[k, 0] / grid_size
                h = anchor_sizes[k, 1] / grid_size
                
                anchors[i, j, k] = torch.tensor([cx, cy, w, h])
    
    return anchors


def calculate_map(predictions, ground_truths, num_classes, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP)
    
    Args:
        predictions: List of predictions for all images
        ground_truths: List of ground truth boxes for all images
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching
        
    Returns:
        float: mAP value
    """
    # Simplified mAP calculation
    # For full implementation, see COCO evaluation metrics
    
    aps = []
    
    for class_id in range(num_classes):
        # Get predictions and ground truths for this class
        class_preds = []
        class_gts = []
        
        for img_preds, img_gts in zip(predictions, ground_truths):
            class_preds.extend([p for p in img_preds if p[5] == class_id])
            class_gts.extend([g for g in img_gts if g[4] == class_id])
        
        if len(class_gts) == 0:
            continue
        
        # Sort predictions by confidence
        class_preds = sorted(class_preds, key=lambda x: x[4], reverse=True)
        
        # Calculate precision and recall
        tp = 0
        fp = 0
        matched_gts = set()
        
        for pred in class_preds:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(class_gts):
                if gt_idx in matched_gts:
                    continue
                
                iou = calculate_iou(pred[:4], gt[:4], box_format='xyxy')
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp += 1
                matched_gts.add(best_gt_idx)
            else:
                fp += 1
        
        # Calculate AP for this class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / len(class_gts) if len(class_gts) > 0 else 0
        
        # Simplified AP (area under precision-recall curve)
        ap = precision * recall
        aps.append(ap)
    
    # Calculate mAP
    mAP = np.mean(aps) if len(aps) > 0 else 0
    
    return mAP


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Detection Utilities Testing")
    print("=" * 70)
    
    # Test IoU calculation
    box1 = [10, 10, 50, 50]  # x1, y1, x2, y2
    box2 = [30, 30, 70, 70]
    iou = calculate_iou(box1, box2, box_format='xyxy')
    print(f"\n✓ IoU between box1 and box2: {iou:.3f}")
    
    # Test NMS
    detections = [
        [10, 10, 50, 50, 0.9, 0],  # High confidence
        [12, 12, 52, 52, 0.8, 0],  # Overlapping, lower confidence
        [100, 100, 150, 150, 0.85, 1],  # Different location
    ]
    filtered = non_maximum_suppression(detections, conf_threshold=0.5, iou_threshold=0.4)
    print(f"✓ NMS: {len(detections)} detections → {len(filtered)} after NMS")
    
    # Test coordinate conversion
    box_xywh = [30, 30, 40, 40]  # center_x, center_y, width, height
    box_xyxy = xywh_to_xyxy(box_xywh)
    print(f"✓ XYWH {box_xywh} → XYXY {box_xyxy}")
    
    # Test anchor generation
    anchors = generate_anchors(grid_size=13, num_anchors=3)
    print(f"✓ Generated anchors: shape {anchors.shape}")
    
    print("\n" + "=" * 70)
