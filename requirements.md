# Requirements
This project was developed using Python and TensorFlow (Keras API) for deep learning experimentation, optimization, and evaluation on the CIFAR-10 dataset.

## Core Framework
- Python 3.9+
- TensorFlow 2.x
- Keras (via TensorFlow)

## Deep Learning & Optimization
- tensorflow.keras
  - Models (Sequential, Model)
  - Layers (Conv2D, DepthwiseConv2D, Dense, Flatten, Dropout, BatchNormalization, ReLU, MaxPooling2D, GlobalAveragePooling2D)
  - Optimizers (Adam)
  - Regularizers (L2)
  - Callbacks (EarlyStopping, ReduceLROnPlateau)
  - Utilities (to_categorical)
  - ImageDataGenerator (Data Augmentation)

- Keras Tuner
  - keras-tuner
  - RandomSearch

## Data Processing & Numerical Computation
- NumPy
- Pandas
- Pickle
- OS
- Tarfile
- Warnings

## Visualization
- Matplotlib
- Seaborn
- Plotly

## Model Evaluation
- Scikit-learn
  - classification_report
  - confusion_matrix
  - ConfusionMatrixDisplay

## Installation
You may install the required libraries using:

```bash
pip install tensorflow keras-tuner numpy pandas matplotlib seaborn plotly scikit-learn
