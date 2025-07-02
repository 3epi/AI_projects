# AI Course Projects - CA4: Convolutional vs. Fully Connected Neural Networks

## Overview

This repository contains the implementation for Computer Assignment 4 (CA4) of the AI course, focusing on comparing Convolutional Neural Networks (CNNs) with Fully Connected Neural Networks (FCNNs) on the CIFAR-10 image classification dataset. The project demonstrates the superiority of convolutional architectures for image recognition tasks.

## Project Structure

```
CA4/
├── AI-CA4-Bazargan-810102303.ipynb      # Main implementation notebook
├── AI_S04_CA4.ipynb                     # Original assignment template
├── PyTorch-tutorial.ipynb               # PyTorch tutorial notebook
├── cnn.pth                              # Trained CNN model weights
├── fully-connected.pth                  # Trained FCNN model weights
└── Readme.md                            # This file
```

## Dataset

The project uses the **CIFAR-10** dataset, which consists of:
- **50,000 training images** (32×32 color images)
- **10,000 test images**
- **10 classes**: plane, car, bird, cat, deer, dog, frog, horse, ship, truck
- **Data split**: 45,000 training, 5,000 validation, 10,000 test

### Data Preprocessing
- **Normalization**: Applied with mean=(0.491, 0.482, 0.446) and std=(0.247, 0.243, 0.261)
- **Tensor conversion**: Images converted to PyTorch tensors
- **Batch processing**: Batch size of 512 for efficient GPU utilization

## Model Architectures

### 1. Fully Connected Neural Network (FCNN)

**Architecture:**
```python
Input: 3×32×32 → Flatten → 3072 features
├── Linear(3072 → 10500) + ReLU + Dropout(0.5)
├── Linear(10500 → 120) + ReLU + Dropout(0.5)
└── Linear(120 → 10) [Output layer]
```

**Key Features:**
- **Parameters**: ~33.5M trainable parameters
- **Regularization**: Dropout layers (0.5) to prevent overfitting
- **Activation**: ReLU activation functions
- **Input processing**: Flattens 2D images to 1D vectors

### 2. Convolutional Neural Network (CNN)

**Architecture:**
```python
Convolutional Feature Extractor:
├── Conv2d(3→64, 3×3) + ReLU + MaxPool2d(2×2)    # 64×16×16
├── Conv2d(64→128, 3×3) + ReLU + MaxPool2d(2×2)  # 128×8×8  
└── Conv2d(128→256, 3×3) + ReLU + MaxPool2d(2×2) # 256×4×4

Feature Space Generator:
├── Flatten → Linear(4096 → 8100) + ReLU + Dropout(0.5)
└── Linear(8100 → 10) [Output layer]
```

**Key Features:**
- **Parameters**: ~33.5M trainable parameters (matched with FCNN)
- **Spatial awareness**: Preserves 2D spatial relationships
- **Feature maps**: Hierarchical feature extraction
- **Pooling**: Progressive dimensionality reduction

## Training Configuration

### Hyperparameters
- **Epochs**: 60 (consistent across both models)
- **Batch size**: 512
- **Loss function**: CrossEntropyLoss
- **Optimizer**: Adam
  - FCNN: Learning rate = 0.0001
  - CNN: Learning rate = 0.001
- **Device**: CUDA (GPU) if available, otherwise CPU

### Training Process
- **Validation monitoring**: Track loss and accuracy per epoch
- **Model saving**: Automatic weight saving after training
- **Overfitting detection**: Training vs validation performance analysis

### Key Findings

#### Why CNN Outperforms FCNN:
1. **Spatial Relationship Preservation**: CNNs maintain 2D spatial structure while FCNNs flatten images to 1D
2. **Local Feature Detection**: Convolutional filters detect edges, textures, and patterns
3. **Translation Invariance**: Pooling layers provide robustness to small translations
4. **Hierarchical Learning**: Progressive feature abstraction from low to high level

#### Overfitting Analysis:
- **FCNN**: Clear overfitting observed (train: 80%, validation: 56%)
- **CNN**: Better generalization with closer train-validation performance

## Advanced Analysis

### 1. Feature Space Exploration
- **Feature extraction**: 8100-dimensional feature vectors before final classification
- **K-NN analysis**: Nearest neighbor search in learned feature space
- **Insight**: Model learns semantic similarities beyond visual appearance

### 2. t-SNE Visualization
- **Dimensionality reduction**: 8100D → 2D using t-SNE
- **Cluster analysis**: Well-separated clusters for distinct classes (trucks, ships)
- **Mixed clusters**: Challenging classes show overlapping representations

### 3. Feature Map Visualization
- **Convolutional filters**: Visualization of learned features in early layers
- **Pattern detection**: Filters learn edge detectors, texture analyzers
- **Hierarchical features**: Low-level to high-level feature progression

### 4. Error Analysis
- **Misclassification visualization**: 24 incorrectly predicted samples
- **Common errors**: Similar-looking classes (cat/dog, bird/plane)
- **Model limitations**: Difficulty with fine-grained distinctions

## Implementation Details

### Key Components
- **Data loading**: [`torchvision.datasets.CIFAR10`](CA4/AI-CA4-Bazargan-810102303.ipynb)
- **Data augmentation**: Normalization transforms
- **Model definition**: Custom [`FullyConnectedNetwork`](CA4/AI-CA4-Bazargan-810102303.ipynb) and [`CNN`](CA4/AI-CA4-Bazargan-810102303.ipynb) classes
- **Training utilities**: [`train_epoch`](CA4/AI-CA4-Bazargan-810102303.ipynb) and [`eval_epoch`](CA4/AI-CA4-Bazargan-810102303.ipynb) functions

### Visualization Features
- **Data visualization**: 5 random samples per class
- **Training curves**: Loss and accuracy plots
- **Unnormalization**: [`UnNormalize`](CA4/AI-CA4-Bazargan-810102303.ipynb) class for display
- **Feature maps**: Intermediate layer visualization

## Key Insights

### Educational Outcomes
1. **Architectural Impact**: CNN's spatial awareness crucial for image tasks
2. **Parameter Efficiency**: Similar parameter count, vastly different performance
3. **Feature Learning**: Hierarchical representation learning in CNNs
4. **Generalization**: Importance of architecture choice for task-specific performance

### Technical Learnings
- **PyTorch fundamentals**: Model definition, training loops, data handling
- **Deep learning concepts**: Convolution, pooling, feature extraction
- **Model analysis**: Overfitting detection, feature space exploration
- **Visualization techniques**: t-SNE, feature maps, error analysis

## Files Description
- [`AI-CA4-Bazargan-810102303.ipynb`](CA4/AI-CA4-Bazargan-810102303.ipynb): Complete implementation and analysis
- [`cnn.pth`](CA4/cnn.pth): Trained CNN model weights
- [`fully-connected.pth`](CA4/fully-connected.pth): Trained FCNN model weights
- [`PyTorch-tutorial.ipynb`](CA4/PyTorch-tutorial.ipynb): PyTorch fundamentals tutorial

**Note**: This project demonstrates fundamental concepts in deep learning and computer vision, providing hands-on experience with PyTorch and neural network architectures.