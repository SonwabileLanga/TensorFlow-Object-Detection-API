# TensorFlow Object Detection API

A comprehensive implementation of the TensorFlow Object Detection API for training custom object detection models. This project provides a complete pipeline from data preparation to model deployment.

![Object Detection](https://user-images.githubusercontent.com/40062143/61365740-81e81500-a888-11e9-8f83-d14f0481025f.jpg)

## 🚀 Features

- **Complete Training Pipeline**: End-to-end training workflow for custom object detection
- **Multiple Model Support**: SSD MobileNet, Faster R-CNN, and other popular architectures
- **Data Processing Tools**: Automated TFRecord generation and dataset splitting
- **Easy Configuration**: Pre-configured templates for different models
- **Inference Scripts**: Ready-to-use inference for images and videos
- **Comprehensive Setup**: Automated environment setup and dependency management

## 📋 Requirements

- Python 3.7 or higher
- TensorFlow 2.10 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM (16GB+ recommended)
- 10GB+ free disk space

## 🛠️ Installation

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SonwabileLanga/TensorFlow-Object-Detection-API.git
   cd TensorFlow-Object-Detection-API
   ```

2. **Run the setup script**:
   ```bash
   python scripts/setup.py
   ```

3. **Activate environment variables**:
   ```bash
   source .env
   ```

### Manual Setup

If you prefer manual setup:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Protocol Buffers**:
   ```bash
   # macOS
   brew install protobuf
   
   # Ubuntu/Debian
   sudo apt-get install protobuf-compiler
   ```

3. **Clone TensorFlow Models**:
   ```bash
   git clone https://github.com/tensorflow/models.git
   cd models/research/object_detection
   protoc object_detection/protos/*.proto --python_out=.
   pip install -e .
   ```

## 📁 Project Structure

```
TensorFlow-Object-Detection-API/
├── configs/                 # Model configuration files
│   ├── ssd_mobilenet_v2.config
│   └── faster_rcnn_resnet50.config
├── data/                    # Dataset directory
│   ├── images/             # Training images
│   ├── annotations/        # XML annotations
│   ├── records/            # TFRecord files
│   └── sample/             # Sample data
├── models/                 # TensorFlow Models repository
├── research/               # Research modules
├── scripts/                # Main scripts
│   ├── train.py           # Training script
│   ├── inference.py       # Inference script
│   └── setup.py           # Setup script
├── training/               # Training outputs
├── utils/                  # Utility functions
│   └── data_utils.py      # Data processing utilities
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🎯 Quick Start

### 1. Prepare Your Dataset

Organize your dataset in the following structure:
```
data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── annotations/
    ├── image1.xml
    ├── image2.xml
    └── ...
```

### 2. Generate Label Map

Create a label map from your annotations:
```bash
python utils/data_utils.py --action create_label_map \
    --annotation_dir data/annotations \
    --output_path data/label_map.pbtxt
```

### 3. Split Dataset

Split your dataset into train/validation/test sets:
```bash
python utils/data_utils.py --action split_dataset \
    --image_dir data/images \
    --annotation_dir data/annotations \
    --output_dir data/split \
    --train_ratio 0.8 \
    --val_ratio 0.1
```

### 4. Generate TFRecords

Convert your dataset to TFRecord format:
```bash
python utils/data_utils.py --action generate_tfrecord \
    --image_dir data/split/images/train \
    --annotation_dir data/split/annotations/train \
    --output_path data/records/train.record \
    --label_map_path data/label_map.pbtxt
```

### 5. Configure Training

Update the configuration file:
```bash
# Edit configs/ssd_mobilenet_v2.config
# Update the following paths:
# - fine_tune_checkpoint: "path/to/pretrained/model"
# - label_map_path: "data/label_map.pbtxt"
# - input_path: "data/records/train.record"
# - num_classes: your_number_of_classes
```

### 6. Start Training

Begin training your model:
```bash
python scripts/train.py \
    --config_path configs/ssd_mobilenet_v2.config \
    --model_dir training/ \
    --num_train_steps 50000
```

### 7. Run Inference

Test your trained model:
```bash
python scripts/inference.py \
    --model_path training/exported_model \
    --input_path data/test_image.jpg \
    --output_path results/detected_image.jpg \
    --confidence_threshold 0.5
```

## 🔧 Configuration

### Model Configurations

The project includes pre-configured templates for popular models:

- **SSD MobileNet V2**: Fast inference, good for mobile/edge devices
- **Faster R-CNN ResNet50**: Higher accuracy, requires more resources

### Key Configuration Parameters

- `num_classes`: Number of object classes in your dataset
- `batch_size`: Training batch size (adjust based on GPU memory)
- `learning_rate`: Learning rate for training
- `num_steps`: Total number of training steps
- `fine_tune_checkpoint`: Path to pre-trained model

## 📊 Monitoring Training

### TensorBoard

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir training/
```

### Key Metrics

- **Loss**: Total training loss
- **Localization Loss**: Bounding box regression loss
- **Classification Loss**: Object classification loss
- **mAP**: Mean Average Precision (evaluation metric)

## 🎨 Data Augmentation

The configuration includes several data augmentation options:

- Random horizontal flip
- Random crop
- Random rotation
- Color jittering

## 🚀 Model Export

Export your trained model for inference:
```bash
python scripts/export_model.py \
    --input_type image_tensor \
    --pipeline_config_path configs/ssd_mobilenet_v2.config \
    --trained_checkpoint_dir training/ \
    --output_directory exports/
```

## 📱 Deployment

### Mobile Deployment

For mobile deployment, use TensorFlow Lite:
```bash
python scripts/convert_to_tflite.py \
    --saved_model_dir exports/ \
    --output_file model.tflite
```

### Web Deployment

For web deployment, use TensorFlow.js:
```bash
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    exports/ web_model/
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in config
   - Use smaller input image size

2. **Protobuf Errors**:
   - Recompile protobuf files: `protoc object_detection/protos/*.proto --python_out=.`

3. **Import Errors**:
   - Check PYTHONPATH: `export PYTHONPATH=$PYTHONPATH:$(pwd)/research`

4. **Training Not Starting**:
   - Verify config file paths
   - Check TFRecord files exist
   - Ensure label map is correct

### Performance Tips

- Use GPU for training (10x faster than CPU)
- Enable mixed precision training for faster training
- Use data augmentation to improve model robustness
- Monitor training with TensorBoard

## 📚 Additional Resources

- [TensorFlow Object Detection API Documentation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/)
- [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
- [LabelImg Tool](https://github.com/tzutalin/labelImg) for annotation
- [COCO Dataset](https://cocodataset.org/) for pre-trained models

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the Object Detection API
- Contributors to the TensorFlow Models repository
- The open-source computer vision community

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing [Issues](https://github.com/SonwabileLanga/TensorFlow-Object-Detection-API/issues)
3. Create a new issue with detailed information

---

**Happy Training! 🎉**
