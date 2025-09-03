#!/usr/bin/env python3
"""
Quick Start Example for TensorFlow Object Detection API

This example demonstrates how to use the object detection API
with a simple synthetic dataset.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    print("Creating sample dataset...")
    
    # Create directories
    data_dir = project_root / "data" / "sample"
    images_dir = data_dir / "images"
    annotations_dir = data_dir / "annotations"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample images with rectangles
    for i in range(10):
        # Create a random image
        image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        
        # Add a colored rectangle (simulating an object)
        x1, y1 = np.random.randint(50, 200, 2)
        x2, y2 = x1 + np.random.randint(50, 100), y1 + np.random.randint(50, 100)
        
        # Draw rectangle
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
        
        # Save image
        image_path = images_dir / f"sample_{i:03d}.jpg"
        cv2.imwrite(str(image_path), image)
        
        # Create corresponding annotation
        annotation_path = annotations_dir / f"sample_{i:03d}.xml"
        create_pascal_voc_annotation(annotation_path, f"sample_{i:03d}.jpg", 
                                   image.shape[1], image.shape[0], 
                                   [(x1, y1, x2, y2, "object")])
    
    print(f"Created {len(list(images_dir.glob('*.jpg')))} sample images")
    print(f"Created {len(list(annotations_dir.glob('*.xml')))} sample annotations")

def create_pascal_voc_annotation(xml_path, filename, width, height, objects):
    """Create a Pascal VOC XML annotation file."""
    from xml.etree.ElementTree import Element, SubElement, tostring
    from xml.dom import minidom
    
    # Create root element
    annotation = Element('annotation')
    
    # Add filename
    SubElement(annotation, 'filename').text = filename
    
    # Add source
    source = SubElement(annotation, 'source')
    SubElement(source, 'database').text = 'Sample Database'
    
    # Add size
    size = SubElement(annotation, 'size')
    SubElement(size, 'width').text = str(width)
    SubElement(size, 'height').text = str(height)
    SubElement(size, 'depth').text = '3'
    
    # Add segmented
    SubElement(annotation, 'segmented').text = '0'
    
    # Add objects
    for x1, y1, x2, y2, class_name in objects:
        obj = SubElement(annotation, 'object')
        SubElement(obj, 'name').text = class_name
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'
        
        bndbox = SubElement(obj, 'bndbox')
        SubElement(bndbox, 'xmin').text = str(x1)
        SubElement(bndbox, 'ymin').text = str(y1)
        SubElement(bndbox, 'xmax').text = str(x2)
        SubElement(bndbox, 'ymax').text = str(y2)
    
    # Write to file
    with open(xml_path, 'w') as f:
        f.write(prettify(annotation))

def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def create_label_map():
    """Create a simple label map."""
    label_map_path = project_root / "data" / "sample" / "label_map.pbtxt"
    
    with open(label_map_path, 'w') as f:
        f.write("item {\n")
        f.write("  name: 'object'\n")
        f.write("  id: 1\n")
        f.write("}\n")
    
    print(f"Created label map: {label_map_path}")

def main():
    """Main function to run the quick start example."""
    print("🚀 TensorFlow Object Detection API - Quick Start Example")
    print("=" * 60)
    
    # Create sample dataset
    create_sample_dataset()
    
    # Create label map
    create_label_map()
    
    print("\n✅ Sample dataset created successfully!")
    print("\nNext steps:")
    print("1. Generate TFRecords:")
    print("   python utils/data_utils.py --action generate_tfrecord \\")
    print("       --image_dir data/sample/images \\")
    print("       --annotation_dir data/sample/annotations \\")
    print("       --output_path data/sample/train.record \\")
    print("       --label_map_path data/sample/label_map.pbtxt")
    print("\n2. Update config file with correct paths")
    print("\n3. Start training:")
    print("   python scripts/train.py --config_path configs/ssd_mobilenet_v2.config")
    
    print("\n🎉 Quick start example completed!")

if __name__ == '__main__':
    main()
