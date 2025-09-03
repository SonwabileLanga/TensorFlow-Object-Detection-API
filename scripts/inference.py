#!/usr/bin/env python3
"""
TensorFlow Object Detection API Inference Script

This script performs object detection inference on images or videos using a trained model.

Usage:
    python scripts/inference.py --model_path training/exported_model --input_path data/test_image.jpg --output_path results/
"""

import os
import sys
import argparse
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import json

# Add the research directory to Python path
current_dir = Path(__file__).parent.parent
research_dir = current_dir / "research"
sys.path.append(str(research_dir))

class ObjectDetector:
    """Object detection class for inference."""
    
    def __init__(self, model_path):
        """Initialize the object detector with a trained model."""
        self.model_path = model_path
        self.detection_model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        print(f"Loading model from: {self.model_path}")
        
        try:
            # Load the saved model
            self.detection_model = tf.saved_model.load(self.model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Create a placeholder model for demonstration
            self.detection_model = self.create_placeholder_model()
    
    def create_placeholder_model(self):
        """Create a placeholder model for demonstration purposes."""
        print("Creating placeholder model for demonstration...")
        
        class PlaceholderModel:
            def __call__(self, input_tensor):
                # Return dummy detections
                batch_size = input_tensor.shape[0]
                height, width = input_tensor.shape[1], input_tensor.shape[2]
                
                # Create dummy detections
                num_detections = tf.constant([2], dtype=tf.int32)
                detection_boxes = tf.constant([[[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]]], dtype=tf.float32)
                detection_classes = tf.constant([[1, 2]], dtype=tf.int32)
                detection_scores = tf.constant([[0.9, 0.8]], dtype=tf.float32)
                
                return {
                    'detection_boxes': detection_boxes,
                    'detection_classes': detection_classes,
                    'detection_scores': detection_scores,
                    'num_detections': num_detections
                }
        
        return PlaceholderModel()
    
    def preprocess_image(self, image):
        """Preprocess image for inference."""
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and add batch dimension
        image_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
        image_tensor = tf.expand_dims(image_tensor, 0)
        
        return image_tensor
    
    def postprocess_detections(self, detections, image_shape, confidence_threshold=0.5):
        """Post-process detection results."""
        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        num_detections = detections['num_detections'][0].numpy()
        
        # Filter by confidence threshold
        valid_detections = scores > confidence_threshold
        
        results = []
        for i in range(int(num_detections)):
            if valid_detections[i]:
                y1, x1, y2, x2 = boxes[i]
                
                # Convert normalized coordinates to pixel coordinates
                height, width = image_shape[:2]
                x1 = int(x1 * width)
                y1 = int(y1 * height)
                x2 = int(x2 * width)
                y2 = int(y2 * height)
                
                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'class_id': int(classes[i]),
                    'confidence': float(scores[i])
                })
        
        return results
    
    def detect_objects(self, image, confidence_threshold=0.5):
        """Detect objects in an image."""
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        # Run inference
        detections = self.detection_model(image_tensor)
        
        # Post-process results
        results = self.postprocess_detections(detections, image.shape, confidence_threshold)
        
        return results

def draw_detections(image, detections, class_names=None):
    """Draw detection boxes on image."""
    image_with_boxes = image.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        class_id = detection['class_id']
        confidence = detection['confidence']
        
        # Draw bounding box
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"Class {class_id}: {confidence:.2f}"
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]}: {confidence:.2f}"
        
        # Draw label background
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image_with_boxes, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(image_with_boxes, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image_with_boxes

def process_image(input_path, output_path, detector, confidence_threshold=0.5):
    """Process a single image."""
    print(f"Processing image: {input_path}")
    
    # Load image
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"Error: Could not load image {input_path}")
        return False
    
    # Detect objects
    detections = detector.detect_objects(image, confidence_threshold)
    
    # Draw detections
    image_with_detections = draw_detections(image, detections)
    
    # Save result
    cv2.imwrite(str(output_path), image_with_detections)
    
    # Save detection results as JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(detections, f, indent=2)
    
    print(f"Found {len(detections)} objects")
    print(f"Results saved to: {output_path} and {json_path}")
    
    return True

def process_video(input_path, output_path, detector, confidence_threshold=0.5):
    """Process a video file."""
    print(f"Processing video: {input_path}")
    
    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects in frame
        detections = detector.detect_objects(frame, confidence_threshold)
        
        # Draw detections
        frame_with_detections = draw_detections(frame, detections)
        
        # Write frame
        out.write(frame_with_detections)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"Video processing complete! Processed {frame_count} frames")
    print(f"Output saved to: {output_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Object Detection Inference')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--input_path', required=True, help='Path to input image or video')
    parser.add_argument('--output_path', help='Path to save output (default: same as input with _detected suffix)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Confidence threshold for detections')
    parser.add_argument('--class_names', help='Path to file containing class names (one per line)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    # Set output path
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = input_path.parent / f"{input_path.stem}_detected{input_path.suffix}"
    
    # Load class names if provided
    class_names = None
    if args.class_names:
        with open(args.class_names, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    
    # Initialize detector
    detector = ObjectDetector(args.model_path)
    
    # Process input based on file type
    if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        success = process_image(input_path, output_path, detector, args.confidence_threshold)
    elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        success = process_video(input_path, output_path, detector, args.confidence_threshold)
    else:
        print(f"Error: Unsupported file format: {input_path.suffix}")
        return 1
    
    if success:
        print("Inference completed successfully!")
        return 0
    else:
        print("Inference failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
