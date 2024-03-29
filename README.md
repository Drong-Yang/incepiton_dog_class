# Dog Breed Classification

Deep learning-based dog breed classification using Inception-v3 architecture.

## Features
- Inception-v3 model trained on 120 dog breeds
- ResNet-50 baseline for comparison
- Interactive GUI for real-time classification
- Data augmentation and preprocessing pipeline
- Early stopping and learning rate scheduling
- Training visualization and confusion matrix

## Project Structure
```
incepiton_dog_class/
├── inception.py        # Inception-v3 model implementation
├── resnet.py           # ResNet-50 baseline model
├── net.py              # Neural network utilities
├── net_resnet.py       # ResNet inference module
├── makedata.py         # Data preprocessing pipeline
├── model.py            # Model architecture definitions
├── train.py            # Training script with config
├── utils_train.py      # Training visualization utilities
├── gui_test.py         # GUI application (main)
├── gui_test_1.py       # GUI with model selection
├── gui.py              # GUI base framework
├── test_model.py       # Inference smoke test
├── preprocess.py       # Dataset and transforms
├── generate_picture.py # Result visualization
├── label.csv           # 120 breed class labels
├── imgs/               # UI assets and icons
└── *.png/*.jpg         # Training results and figures
```

## Usage
```bash
pip install -r requirements.txt
python gui_test.py
```

## Model Performance
- Inception-v3: ~85% top-1 accuracy on test set
- ResNet-50: ~82% top-1 accuracy on test set

## Development
Graduation project, Feb 27 - Mar 29, 2024
