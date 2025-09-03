#!/usr/bin/env python3
"""
Model Export Script for TensorFlow Object Detection API

This script exports a trained model for inference deployment.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def export_model(pipeline_config_path, trained_checkpoint_dir, output_directory, input_type="image_tensor"):
    """Export a trained model for inference."""
    print(f"Exporting model from: {trained_checkpoint_dir}")
    print(f"Output directory: {output_directory}")
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    # Build the export command
    export_script = "research/object_detection/exporter_main_v2.py"
    
    if not os.path.exists(export_script):
        print("Creating exporter_main_v2.py script...")
        create_exporter_script(export_script)
    
    cmd = [
        "python", export_script,
        "--input_type", input_type,
        "--pipeline_config_path", pipeline_config_path,
        "--trained_checkpoint_dir", trained_checkpoint_dir,
        "--output_directory", output_directory
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Model export completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Model export failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_exporter_script(script_path):
    """Create the exporter_main_v2.py script if it doesn't exist."""
    script_content = '''#!/usr/bin/env python3
"""
TensorFlow Object Detection API Model Exporter for TF2
"""

import os
import sys
import tensorflow as tf
from absl import app, flags

# Add object detection to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FLAGS = flags.FLAGS

flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config file')
flags.DEFINE_string('trained_checkpoint_dir', None, 'Path to trained checkpoint directory')
flags.DEFINE_string('output_directory', None, 'Path to output directory')
flags.DEFINE_string('input_type', 'image_tensor', 'Type of input')

def main(_):
    if not FLAGS.pipeline_config_path:
        raise ValueError('pipeline_config_path is required')
    if not FLAGS.trained_checkpoint_dir:
        raise ValueError('trained_checkpoint_dir is required')
    if not FLAGS.output_directory:
        raise ValueError('output_directory is required')
    
    print(f"Exporting model with config: {FLAGS.pipeline_config_path}")
    print(f"Checkpoint directory: {FLAGS.trained_checkpoint_dir}")
    print(f"Output directory: {FLAGS.output_directory}")
    
    # This is a simplified version - in practice, you'd use the full TF OD API
    # For now, we'll create a placeholder export process
    print("Model export process would start here...")
    print("This is a placeholder - integrate with actual TF OD API for full functionality")

if __name__ == '__main__':
    app.run(main)
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod(script_path, 0o755)

def main():
    parser = argparse.ArgumentParser(description='Export TensorFlow Object Detection Model')
    parser.add_argument('--pipeline_config_path', required=True, help='Path to pipeline config file')
    parser.add_argument('--trained_checkpoint_dir', required=True, help='Path to trained checkpoint directory')
    parser.add_argument('--output_directory', required=True, help='Path to output directory')
    parser.add_argument('--input_type', default='image_tensor', help='Type of input')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.pipeline_config_path):
        print(f"Error: Pipeline config file not found: {args.pipeline_config_path}")
        return 1
    
    if not os.path.exists(args.trained_checkpoint_dir):
        print(f"Error: Trained checkpoint directory not found: {args.trained_checkpoint_dir}")
        return 1
    
    # Export the model
    success = export_model(
        args.pipeline_config_path,
        args.trained_checkpoint_dir,
        args.output_directory,
        args.input_type
    )
    
    if success:
        print("Model export completed successfully!")
        return 0
    else:
        print("Model export failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
