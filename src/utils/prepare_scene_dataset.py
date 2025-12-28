"""
Dataset Preparation Script for Scene Classification
Downloads and organizes images for 10 scene classes
"""

import os
import requests
from pathlib import Path
import shutil
from tqdm import tqdm
import argparse


# Scene classes for the project
SCENE_CLASSES = [
    'beach', 'classroom', 'forest', 'gym', 'kitchen',
    'living_room', 'mountain', 'office', 'park', 'street'
]


def create_directory_structure(base_dir):
    """
    Create directory structure for dataset
    
    Structure:
    base_dir/
        train/
            class1/
            class2/
            ...
        val/
            class1/
            class2/
            ...
        test/
            class1/
            class2/
            ...
    """
    base_path = Path(base_dir)
    
    for split in ['train', 'val', 'test']:
        for class_name in SCENE_CLASSES:
            class_dir = base_path / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created directory structure at {base_dir}")
    return base_path


def download_from_existing_dataset(base_dir):
    """
    Copy images from existing dataset if available
    """
    # Check if there's an existing dataset
    existing_dataset = Path('dataset')
    
    if not existing_dataset.exists():
        print("No existing dataset found. Please manually add images or use another source.")
        return False
    
    print("Checking for existing images...")
    
    # Look for images in the dataset directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    found_images = False
    
    for class_name in SCENE_CLASSES:
        # Search for images in various possible locations
        possible_paths = [
            existing_dataset / class_name,
            existing_dataset / 'train' / class_name,
            existing_dataset / 'images' / class_name,
        ]
        
        for path in possible_paths:
            if path.exists():
                images = [f for f in path.iterdir() 
                         if f.suffix.lower() in image_extensions]
                
                if images:
                    print(f"  Found {len(images)} images for {class_name}")
                    found_images = True
                    
                    # Split into train/val/test (70/15/15)
                    num_images = len(images)
                    train_split = int(0.7 * num_images)
                    val_split = int(0.85 * num_images)
                    
                    # Copy to train
                    for img in images[:train_split]:
                        dst = Path(base_dir) / 'train' / class_name / img.name
                        shutil.copy(img, dst)
                    
                    # Copy to val
                    for img in images[train_split:val_split]:
                        dst = Path(base_dir) / 'val' / class_name / img.name
                        shutil.copy(img, dst)
                    
                    # Copy to test
                    for img in images[val_split:]:
                        dst = Path(base_dir) / 'test' / class_name / img.name
                        shutil.copy(img, dst)
                    
                    break
    
    return found_images


def print_dataset_summary(base_dir):
    """Print summary of dataset"""
    base_path = Path(base_dir)
    
    print("\n" + "=" * 70)
    print("Dataset Summary")
    print("=" * 70)
    
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        total = 0
        for class_name in SCENE_CLASSES:
            class_dir = base_path / split / class_name
            if class_dir.exists():
                num_images = len(list(class_dir.glob('*')))
                total += num_images
                print(f"  {class_name:15s}: {num_images:4d} images")
        print(f"  {'TOTAL':15s}: {total:4d} images")
    
    print("\n" + "=" * 70)


def main(args):
    """Main function"""
    print("=" * 70)
    print("Scene Classification Dataset Preparation")
    print("=" * 70)
    
    # Create directory structure
    print("\nCreating directory structure...")
    base_dir = create_directory_structure(args.output_dir)
    
    # Try to use existing dataset
    print("\nLooking for existing images...")
    found = download_from_existing_dataset(args.output_dir)
    
    if not found:
        print("\n" + "=" * 70)
        print("MANUAL DATASET SETUP REQUIRED")
        print("=" * 70)
        print("\nNo existing images found. Please add images manually:")
        print(f"\n1. Download images for each class:")
        for class_name in SCENE_CLASSES:
            print(f"   - {class_name}")
        
        print(f"\n2. Organize them in the following structure:")
        print(f"   {args.output_dir}/")
        print(f"     train/")
        print(f"       beach/")
        print(f"       classroom/")
        print(f"       ...")
        print(f"     val/")
        print(f"       beach/")
        print(f"       ...")
        print(f"     test/")
        print(f"       beach/")
        print(f"       ...")
        
        print(f"\n3. Recommended: 100-200 images per class")
        print(f"   - Train: 70% of images")
        print(f"   - Val: 15% of images")
        print(f"   - Test: 15% of images")
        
        print("\nSuggested image sources:")
        print("  - Google Images (manual download)")
        print("  - Unsplash (https://unsplash.com/)")
        print("  - Pexels (https://www.pexels.com/)")
        print("  - Places365 dataset (http://places2.csail.mit.edu/)")
        
        return
    
    # Print summary
    print_dataset_summary(args.output_dir)
    
    print("\n✓ Dataset preparation complete!")
    print(f"✓ Dataset location: {args.output_dir}")
    print("\nYou can now train the model using:")
    print(f"  python src/training/train_vgg_classifier.py --data-dir {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare scene classification dataset')
    parser.add_argument('--output-dir', type=str, default='dataset/scenes',
                       help='Output directory for organized dataset')
    
    args = parser.parse_args()
    main(args)
