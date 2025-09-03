#!/usr/bin/env python3
"""
Setup script for TensorFlow Object Detection API

This script sets up the environment, installs dependencies, and configures
the project for object detection training and inference.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and handle errors."""
    print(f"Running: {description or command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description or command} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running: {description or command}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"✗ Python {version.major}.{version.minor} is not supported. Please use Python 3.7 or higher.")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    # Upgrade pip
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def setup_protobuf():
    """Set up Protocol Buffers."""
    print("Setting up Protocol Buffers...")
    
    # Check if protoc is installed
    try:
        subprocess.run(["protoc", "--version"], check=True, capture_output=True)
        print("✓ Protocol Buffers already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing Protocol Buffers...")
        
        # Install protobuf compiler
        if sys.platform == "darwin":  # macOS
            if not run_command("brew install protobuf", "Installing protobuf via Homebrew"):
                print("Please install Homebrew first: https://brew.sh/")
                return False
        elif sys.platform == "linux":  # Linux
            if not run_command("sudo apt-get update && sudo apt-get install -y protobuf-compiler", 
                             "Installing protobuf via apt"):
                return False
        else:
            print("Please install Protocol Buffers manually for your platform")
            return False
    
    return True

def setup_tensorflow_models():
    """Set up TensorFlow Models repository."""
    print("Setting up TensorFlow Models...")
    
    models_dir = Path("models")
    if models_dir.exists():
        print("✓ TensorFlow Models directory already exists")
        return True
    
    # Clone TensorFlow Models repository
    if not run_command("git clone https://github.com/tensorflow/models.git", 
                      "Cloning TensorFlow Models repository"):
        return False
    
    # Install object detection API
    object_detection_dir = models_dir / "research" / "object_detection"
    if object_detection_dir.exists():
        os.chdir(object_detection_dir)
        
        # Compile protobuf files
        if not run_command("protoc object_detection/protos/*.proto --python_out=.", 
                          "Compiling protobuf files"):
            return False
        
        # Install object detection package
        if not run_command("pip install -e .", "Installing object detection package"):
            return False
        
        os.chdir("../..")
    
    return True

def setup_environment_variables():
    """Set up environment variables."""
    print("Setting up environment variables...")
    
    current_dir = Path.cwd()
    research_dir = current_dir / "research"
    
    # Add to PYTHONPATH
    pythonpath = str(research_dir)
    if "PYTHONPATH" in os.environ:
        pythonpath = f"{pythonpath}:{os.environ['PYTHONPATH']}"
    
    os.environ["PYTHONPATH"] = pythonpath
    
    # Create .env file
    env_content = f"""# TensorFlow Object Detection API Environment Variables
export PYTHONPATH="{pythonpath}"
export TF_CPP_MIN_LOG_LEVEL=2
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✓ Environment variables configured")
    print("To activate environment variables, run: source .env")
    
    return True

def create_directories():
    """Create necessary directories."""
    print("Creating project directories...")
    
    directories = [
        "data/images",
        "data/annotations", 
        "data/records",
        "training",
        "models",
        "exports",
        "logs",
        "results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def download_sample_data():
    """Download sample data for testing."""
    print("Downloading sample data...")
    
    # Create sample images directory
    sample_dir = Path("data/sample")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple test script
    test_script = """#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import cv2

def create_sample_image():
    # Create a simple test image
    image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    cv2.imwrite('data/sample/test_image.jpg', image)
    print("Sample image created: data/sample/test_image.jpg")

if __name__ == "__main__":
    create_sample_image()
"""
    
    with open("scripts/create_sample_data.py", "w") as f:
        f.write(test_script)
    
    # Run the script
    if run_command("python scripts/create_sample_data.py", "Creating sample data"):
        print("✓ Sample data created")
        return True
    
    return False

def test_installation():
    """Test the installation."""
    print("Testing installation...")
    
    test_script = """#!/usr/bin/env python3
import sys
import os

# Add research to path
sys.path.append('research')

try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__} imported successfully")
    
    # Test basic TensorFlow operations
    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5, 6])
    c = tf.add(a, b)
    print(f"✓ TensorFlow operations working: {c.numpy()}")
    
    print("✓ Installation test passed!")
    return True
    
except Exception as e:
    print(f"✗ Installation test failed: {e}")
    return False
"""
    
    with open("test_installation.py", "w") as f:
        f.write(test_script)
    
    success = run_command("python test_installation.py", "Testing installation")
    
    # Clean up test file
    if os.path.exists("test_installation.py"):
        os.remove("test_installation.py")
    
    return success

def main():
    parser = argparse.ArgumentParser(description='Setup TensorFlow Object Detection API')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
    parser.add_argument('--skip-models', action='store_true', help='Skip TensorFlow Models setup')
    parser.add_argument('--skip-test', action='store_true', help='Skip installation test')
    
    args = parser.parse_args()
    
    print("🚀 Setting up TensorFlow Object Detection API...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create directories
    if not create_directories():
        return 1
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            return 1
    
    # Setup Protocol Buffers
    if not setup_protobuf():
        return 1
    
    # Setup TensorFlow Models
    if not args.skip_models:
        if not setup_tensorflow_models():
            return 1
    
    # Setup environment variables
    if not setup_environment_variables():
        return 1
    
    # Download sample data
    if not download_sample_data():
        return 1
    
    # Test installation
    if not args.skip_test:
        if not test_installation():
            return 1
    
    print("=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate environment: source .env")
    print("2. Prepare your dataset in data/images/ and data/annotations/")
    print("3. Generate TFRecords: python utils/data_utils.py --action generate_tfrecord")
    print("4. Update config files in configs/")
    print("5. Start training: python scripts/train.py --config_path configs/ssd_mobilenet_v2.config")
    print("\nFor more information, see README.md")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
