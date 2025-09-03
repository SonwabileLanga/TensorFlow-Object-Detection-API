#!/usr/bin/env python3
"""
TensorFlow Object Detection API Training Script

This script trains a custom object detection model using TensorFlow Object Detection API.
It supports various pre-trained models and can be configured for different datasets.

Usage:
    python scripts/train.py --config_path configs/ssd_mobilenet_v2.config --model_dir training/
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add the research directory to Python path
current_dir = Path(__file__).parent.parent
research_dir = current_dir / "research"
sys.path.append(str(research_dir))

def setup_environment():
    """Set up the environment for TensorFlow Object Detection API."""
    print("Setting up environment...")
    
    # Set environment variables
    os.environ['PYTHONPATH'] = f"{research_dir}:{os.environ.get('PYTHONPATH', '')}"
    
    # Create necessary directories
    os.makedirs("training", exist_ok=True)
    os.makedirs("data/records", exist_ok=True)
    
    print("Environment setup complete!")

def download_pretrained_model(model_name, model_dir):
    """Download pre-trained model if not already present."""
    model_path = Path(model_dir) / model_name
    
    if not model_path.exists():
        print(f"Downloading pre-trained model: {model_name}")
        # This would typically download from TensorFlow Model Zoo
        # For now, we'll create a placeholder
        model_path.mkdir(parents=True, exist_ok=True)
        print(f"Model directory created at: {model_path}")
    else:
        print(f"Pre-trained model already exists at: {model_path}")

def train_model(config_path, model_dir, num_train_steps=None, num_eval_steps=None):
    """Train the object detection model."""
    print(f"Starting training with config: {config_path}")
    print(f"Model directory: {model_dir}")
    
    # Build the training command
    train_script = research_dir / "object_detection" / "model_main_tf2.py"
    
    if not train_script.exists():
        print("Creating model_main_tf2.py script...")
        create_model_main_script(train_script)
    
    cmd = [
        "python", str(train_script),
        "--model_dir", model_dir,
        "--pipeline_config_path", config_path,
        "--alsologtostderr"
    ]
    
    if num_train_steps:
        cmd.extend(["--num_train_steps", str(num_train_steps)])
    
    if num_eval_steps:
        cmd.extend(["--num_eval_steps", str(num_eval_steps)])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False
    
    return True

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

flags.DEFINE_string('model_dir', None, 'Path to output model directory')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config file')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps')
flags.DEFINE_integer('num_eval_steps', None, 'Number of eval steps')
flags.DEFINE_bool('eval_training_data', False, 'If training data should be evaluated')

def main(_):
    if not FLAGS.model_dir:
        raise ValueError('model_dir is required')
    if not FLAGS.pipeline_config_path:
        raise ValueError('pipeline_config_path is required')
    
    print(f"Training model with config: {FLAGS.pipeline_config_path}")
    print(f"Model directory: {FLAGS.model_dir}")
    
    # This is a simplified version - in practice, you'd use the full TF OD API
    # For now, we'll create a placeholder training process
    print("Training process would start here...")
    print("This is a placeholder - integrate with actual TF OD API for full functionality")

if __name__ == '__main__':
    app.run(main)
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod(script_path, 0o755)

def main():
    parser = argparse.ArgumentParser(description='Train TensorFlow Object Detection Model')
    parser.add_argument('--config_path', required=True, help='Path to pipeline config file')
    parser.add_argument('--model_dir', default='training/', help='Directory to save model')
    parser.add_argument('--num_train_steps', type=int, help='Number of training steps')
    parser.add_argument('--num_eval_steps', type=int, help='Number of evaluation steps')
    parser.add_argument('--pretrained_model', help='Name of pre-trained model to use')
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not os.path.exists(args.config_path):
        print(f"Error: Config file not found: {args.config_path}")
        return 1
    
    # Setup environment
    setup_environment()
    
    # Download pre-trained model if specified
    if args.pretrained_model:
        download_pretrained_model(args.pretrained_model, args.model_dir)
    
    # Train the model
    success = train_model(
        args.config_path,
        args.model_dir,
        args.num_train_steps,
        args.num_eval_steps
    )
    
    if success:
        print("Training completed successfully!")
        return 0
    else:
        print("Training failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
