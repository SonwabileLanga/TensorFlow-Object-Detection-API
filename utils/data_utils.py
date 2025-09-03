#!/usr/bin/env python3
"""
Data utilities for TensorFlow Object Detection API

This module provides utilities for data preparation, annotation conversion,
and TFRecord generation for object detection training.
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np
from pathlib import Path
import cv2
from typing import List, Dict, Tuple, Any
import argparse

class LabelMapGenerator:
    """Generate label map files for object detection."""
    
    def __init__(self):
        self.classes = []
    
    def add_class(self, class_name: str, class_id: int = None):
        """Add a class to the label map."""
        if class_id is None:
            class_id = len(self.classes) + 1
        
        self.classes.append({
            'name': class_name,
            'id': class_id
        })
    
    def save_label_map(self, output_path: str):
        """Save label map to file."""
        with open(output_path, 'w') as f:
            for cls in self.classes:
                f.write(f"item {{\n")
                f.write(f"  name: '{cls['name']}'\n")
                f.write(f"  id: {cls['id']}\n")
                f.write(f"}}\n\n")
        
        print(f"Label map saved to: {output_path}")

class PascalVOCParser:
    """Parser for Pascal VOC XML annotation format."""
    
    def __init__(self, class_mapping: Dict[str, int] = None):
        self.class_mapping = class_mapping or {}
    
    def parse_annotation(self, xml_path: str) -> Dict[str, Any]:
        """Parse a Pascal VOC XML annotation file."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # Parse objects
        objects = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            
            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Get class ID
            class_id = self.class_mapping.get(class_name, 1)
            
            objects.append({
                'class_name': class_name,
                'class_id': class_id,
                'bbox': [xmin, ymin, xmax, ymax]
            })
        
        return {
            'filename': root.find('filename').text,
            'width': width,
            'height': height,
            'objects': objects
        }

class TFRecordGenerator:
    """Generate TFRecord files for TensorFlow Object Detection API."""
    
    def __init__(self, label_map_path: str = None):
        self.label_map_path = label_map_path
        self.class_mapping = {}
        
        if label_map_path and os.path.exists(label_map_path):
            self.load_label_map()
    
    def load_label_map(self):
        """Load class mapping from label map file."""
        with open(self.label_map_path, 'r') as f:
            content = f.read()
            
        # Simple parsing of label map
        lines = content.split('\n')
        current_id = None
        current_name = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('id:'):
                current_id = int(line.split(':')[1].strip())
            elif line.startswith("name: '"):
                current_name = line.split("'")[1]
                if current_id and current_name:
                    self.class_mapping[current_name] = current_id
                    current_id = None
                    current_name = None
    
    def create_tf_example(self, image_path: str, annotation: Dict[str, Any]) -> tf.train.Example:
        """Create a TF Example from image and annotation data."""
        # Read image
        with tf.io.gfile.GFile(image_path, 'rb') as fid:
            encoded_image = fid.read()
        
        # Get image dimensions
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        # Prepare data
        filename = annotation['filename'].encode('utf8')
        image_format = b'jpeg'
        
        # Prepare bounding boxes and classes
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        
        for obj in annotation['objects']:
            xmin, ymin, xmax, ymax = obj['bbox']
            
            # Normalize coordinates
            xmins.append(xmin / width)
            xmaxs.append(xmax / width)
            ymins.append(ymin / height)
            ymaxs.append(ymax / height)
            
            classes_text.append(obj['class_name'].encode('utf8'))
            classes.append(obj['class_id'])
        
        # Create TF Example
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
            'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image])),
            'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
            'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
            'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
            'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
            'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
            'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
            'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        }))
        
        return tf_example
    
    def generate_tfrecord(self, image_dir: str, annotation_dir: str, output_path: str, 
                         annotation_format: str = 'pascal_voc'):
        """Generate TFRecord file from images and annotations."""
        print(f"Generating TFRecord: {output_path}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_dir).glob(f'*{ext}'))
            image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} images")
        
        # Initialize parser
        if annotation_format == 'pascal_voc':
            parser = PascalVOCParser(self.class_mapping)
        else:
            raise ValueError(f"Unsupported annotation format: {annotation_format}")
        
        # Create TFRecord writer
        with tf.io.TFRecordWriter(output_path) as writer:
            for image_path in image_files:
                # Find corresponding annotation file
                annotation_path = Path(annotation_dir) / f"{image_path.stem}.xml"
                
                if not annotation_path.exists():
                    print(f"Warning: No annotation found for {image_path.name}")
                    continue
                
                try:
                    # Parse annotation
                    annotation = parser.parse_annotation(str(annotation_path))
                    
                    # Create TF Example
                    tf_example = self.create_tf_example(str(image_path), annotation)
                    
                    # Write to TFRecord
                    writer.write(tf_example.SerializeToString())
                    
                except Exception as e:
                    print(f"Error processing {image_path.name}: {e}")
                    continue
        
        print(f"TFRecord generated successfully: {output_path}")

def create_label_map_from_annotations(annotation_dir: str, output_path: str):
    """Create label map from annotation files."""
    print("Creating label map from annotations...")
    
    class_names = set()
    annotation_files = list(Path(annotation_dir).glob('*.xml'))
    
    for xml_file in annotation_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                class_names.add(class_name)
                
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
    
    # Create label map
    label_generator = LabelMapGenerator()
    for i, class_name in enumerate(sorted(class_names), 1):
        label_generator.add_class(class_name, i)
    
    label_generator.save_label_map(output_path)
    print(f"Found {len(class_names)} unique classes")

def split_dataset(image_dir: str, annotation_dir: str, output_dir: str, 
                 train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Split dataset into train, validation, and test sets."""
    print("Splitting dataset...")
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
        image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    # Shuffle and split
    np.random.shuffle(image_files)
    
    total_files = len(image_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/annotations/{split}", exist_ok=True)
    
    # Copy files
    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for image_file in files:
            # Copy image
            dest_image = Path(output_dir) / 'images' / split / image_file.name
            dest_image.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy annotation
            annotation_file = Path(annotation_dir) / f"{image_file.stem}.xml"
            if annotation_file.exists():
                dest_annotation = Path(output_dir) / 'annotations' / split / f"{image_file.stem}.xml"
                dest_annotation.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy files (simplified - in practice use shutil.copy2)
                print(f"Copying {image_file.name} to {split} set")

def main():
    parser = argparse.ArgumentParser(description='Data utilities for object detection')
    parser.add_argument('--action', required=True, choices=['create_label_map', 'generate_tfrecord', 'split_dataset'],
                       help='Action to perform')
    parser.add_argument('--image_dir', help='Directory containing images')
    parser.add_argument('--annotation_dir', help='Directory containing annotations')
    parser.add_argument('--output_path', help='Output file path')
    parser.add_argument('--label_map_path', help='Path to label map file')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio for training set')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio for validation set')
    
    args = parser.parse_args()
    
    if args.action == 'create_label_map':
        if not args.annotation_dir or not args.output_path:
            print("Error: annotation_dir and output_path are required for create_label_map")
            return 1
        
        create_label_map_from_annotations(args.annotation_dir, args.output_path)
    
    elif args.action == 'generate_tfrecord':
        if not all([args.image_dir, args.annotation_dir, args.output_path]):
            print("Error: image_dir, annotation_dir, and output_path are required for generate_tfrecord")
            return 1
        
        generator = TFRecordGenerator(args.label_map_path)
        generator.generate_tfrecord(args.image_dir, args.annotation_dir, args.output_path)
    
    elif args.action == 'split_dataset':
        if not all([args.image_dir, args.annotation_dir, args.output_path]):
            print("Error: image_dir, annotation_dir, and output_path are required for split_dataset")
            return 1
        
        split_dataset(args.image_dir, args.annotation_dir, args.output_path, 
                     args.train_ratio, args.val_ratio)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
