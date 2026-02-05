# Optimizing Deep Learning Models for CIFAR-10 Image Classification  
A Comparative Study of 2D CNN and ConvMLP Architectures

## Project Overview
This project presents an end-to-end deep learning study focused on optimizing neural network architectures for image classification using the **CIFAR-10 dataset**.

The primary objective is to identify an **optimal deep learning model** through systematic experimentation, hyperparameter optimization, and model evaluation. Two predictive architectures are implemented and compared:
- **2D Convolutional Neural Network (2D CNN)**
- **Convolutional Multi-Layer Perceptron (ConvMLP)**

The study emphasizes both **model performance** and **optimization strategy**, applying modern deep learning techniques to improve generalization and predictive accuracy.


## Project Objectives
- Perform comprehensive **data preparation and preprocessing**
- Conduct **Exploratory Data Analysis (EDA)** on CIFAR-10
- Design and implement baseline deep learning models
- Apply optimization techniques to improve model performance
- Compare CNN and ConvMLP architectures
- Evaluate performance using suitable classification metrics
- Critically analyze findings in relation to peer-reviewed literature


## Dataset
**Dataset:** CIFAR-10  
**Source:** Department of Computer Science, University of Toronto  
**URL:** https://www.cs.toronto.edu/~kriz/cifar.html  
**Version Used:** CIFAR-10 Python version  

### Dataset Description
CIFAR-10 consists of:
- 60,000 32x32 color images
- 10 object classes
- 50,000 training images
- 10,000 testing images

## Data Preparation & Pre-processing
The project includes:
- Data normalization and scaling
- One-hot encoding of class labels
- Train-validation split
- Data augmentation techniques to improve generalization

### Data Augmentation Techniques
- Random horizontal flipping
- Random rotations
- Image shifting
These techniques help reduce overfitting and improve model robustness.

## Literature Review
A comprehensive literature review was conducted to:

- Analyze previous CNN-based image classification approaches
- Examine optimization strategies for hyperparameter tuning
- Compare traditional MLP models with convolution-based architectures
- Identify best practices in training deep learning models on CIFAR-10

The selection of CIFAR-10 is justified due to:
- Its balanced multi-class structure
- Standardized benchmarking role
- Suitability for evaluating convolutional architectures
- Availability of extensive comparative research


## Model Architectures
### Baseline 2D CNN
- Convolutional layers
- ReLU activation functions
- Max pooling layers
- Fully connected dense layers
- Softmax output layer

### Baseline ConvMLP 
- Convolutional feature extraction
- Flattened intermediate representation
- Deep MLP classifier layers
- Batch normalization
- ReLU activation
- Dropout regularization


## Optimization Techniques Applied
The project incorporates optimization concepts including:
- Data augmentation
- Early stopping
- Model checkpoints
- Callbacks
- Batch normalization
- ReLU activation functions
- Hyperparameter tuning
- Random Search (Keras Tuner)

### Hyperparameters Tuned
- Learning rate
- Batch size
- Number of convolutional filters
- Kernel size
- Number of dense units
- Dropout rate
- Optimizer selection
Random Search and Keras Tuner were used to identify optimal meta-parameters.


## Model Evaluation
The following evaluation metrics are used:
- Accuracy
- Loss
- Precision
- Recall
- F1-score
- Support
- Confusion Matrix

### Visual Analysis Includes:
- Training vs validation accuracy plots
- Training vs validation loss plots
- Classification report
- Confusion matrix visualization


## Experimental Workflow
1. Dataset loading and preprocessing
2. Exploratory Data Analysis (EDA)
3. Baseline model implementation
4. Model training and validation
5. Hyperparameter tuning
6. Optimized model training
7. Performance evaluation
8. Comparative analysis between CNN and ConvMLP


## Results & Comparative Analysis
The final section includes:
- Performance comparison between CNN and ConvMLP
- Evaluation of optimization impact
- Critical analysis against peer-reviewed studies
- Discussion of model strengths and weaknesses
- Identification of potential improvements


## Conclusion
The optimized CNN achieved a test accuracy of 88.46%, while ConvMLP attained a competitive 84.85%, significantly outperforming traditional MLP architectures reported in prior literature.

However, hardware constraints (12GB RAM, no GPU acceleration) limited experimentation with deeper architectures such as EfficientNet or large-scale ensemble methods. Training time and memory usage restricted exploration of larger design spaces.

Future improvements may include:
Transfer learning using pre-trained models (e.g., ResNet-50, EfficientNet)
Advanced data augmentation techniques (e.g., CutMix, FMix)
Bayesian Optimization or Hyperband for more efficient hyperparameter search
Deeper ConvMLP configurations where computational resources allow