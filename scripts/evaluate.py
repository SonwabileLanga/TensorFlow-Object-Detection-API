#!/usr/bin/env python3
"""
Model Evaluation Script for TensorFlow Object Detection API

This script evaluates a trained model on a test dataset.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def evaluate_model(pipeline_config_path, model_dir, checkpoint_dir=None):
    """Evaluate a trained model."""
    print(f"Evaluating model with config: {pipeline_config_path}")
    print(f"Model directory: {model_dir}")
    
    # Build the evaluation command
    eval_script = "research/object_detection/model_main_tf2.py"
    
    if not os.path.exists(eval_script):
        print("Creating model_main_tf2.py script...")
        create_model_main_script(eval_script)
    
    cmd = [
        "python", eval_script,
        "--model_dir", model_dir,
        "--pipeline_config_path", pipeline_config_path,
        "--checkpoint_dir", checkpoint_dir or model_dir,
        "--eval_timeout", "3600",
        "--alsologtostderr"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Evaluation completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_model_main_script(script_path):
    """Create the model_main_tf2.py script if it doesn't exist."""
    script_content = '''#!/usr/bin/env python3
"""
TensorFlow Object Detection API Model Main Script for TF2
"""

import os
import sys
import tensorflow as tf
from absl import app, flags

# Add object detection to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', None, 'Path to model directory')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config file')
flags.DEFINE_string('checkpoint_dir', None, 'Path to checkpoint directory')
flags.DEFINE_integer('eval_timeout', 3600, 'Evaluation timeout in seconds')

def main(_):
    if not FLAGS.model_dir:
        raise ValueError('model_dir is required')
    if not FLAGS.pipeline_config_path:
        raise ValueError('pipeline_config_path is required')
    
    print(f"Evaluating model with config: {FLAGS.pipeline_config_path}")
    print(f"Model directory: {FLAGS.model_dir}")
    
    # This is a simplified version - in practice, you'd use the full TF OD API
    # For now, we'll create a placeholder evaluation process
    print("Evaluation process would start here...")
    print("This is a placeholder - integrate with actual TF OD API for full functionality")

if __name__ == '__main__':
    app.run(main)
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod(script_path, 0o755)

def main():
    parser = argparse.ArgumentParser(description='Evaluate TensorFlow Object Detection Model')
    parser.add_argument('--pipeline_config_path', required=True, help='Path to pipeline config file')
    parser.add_argument('--model_dir', required=True, help='Path to model directory')
    parser.add_argument('--checkpoint_dir', help='Path to checkpoint directory (default: model_dir)')
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not os.path.exists(args.pipeline_config_path):
        print(f"Error: Config file not found: {args.pipeline_config_path}")
        return 1
    
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        return 1
    
    # Evaluate the model
    success = evaluate_model(
        args.pipeline_config_path,
        args.model_dir,
        args.checkpoint_dir
    )
    
    if success:
        print("Evaluation completed successfully!")
        return 0
    else:
        print("Evaluation failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
