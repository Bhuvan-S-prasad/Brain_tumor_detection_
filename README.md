# Brain Tumor Detection Project

This project focuses on the detection and classification of brain tumors from MRI scans using a deep learning model. Leveraging a pretrained ResNet-50 model with weights from IMAGENET1K_v1, the model is fine-tuned to classify images into four categories: **glioma**, **healthy**, **meningioma**, and **pituitary**. The project uses data augmentation, a learning rate scheduler, and early stopping to enhance model performance and avoid overfitting.

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Training Parameters](#training-parameters)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Project Overview

This brain tumor detection project employs deep learning techniques to classify MRI scans. The model is trained on a Kaggle dataset and uses a pretrained ResNet-50 architecture to fine-tune classification accuracy across four distinct tumor classes. A Grad-CAM visualization provides interpretability by highlighting the most influential features in the MRI scans contributing to each classification decision.

## Data

The dataset for this project was sourced from Kaggle and consists of MRI images labeled into four categories:

- **glioma**
- **healthy**
- **meningioma**
- **pituitary**

The data split is as follows:
- **Training set**: 80%
- **Validation set**: 10%
- **Test set**: 10%

## Model Architecture

The model architecture is based on **ResNet-50** with **pretrained weights from IMAGENET1K_v1**. The final layer of the model has been modified to output predictions for four classes.

## Training Parameters

The model was trained with the following parameters:

- **Number of Epochs**: 3
- **Criterion**: CrossEntropyLoss
- **Batch Size**: 32
- **Optimizer**: Adam
  - Learning Rate: 1e-4
  - Weight Decay: 1e-5
- **Scheduler**: ReduceLROnPlateau
  - Mode: Minimize validation loss
  - Factor: 0.5 (learning rate reduces by half if no improvement)
  - Patience: 5 epochs
  - Verbose: True
- **Early Stopping**: Patience of 5 epochs

## Project Structure
```
├── brain.pth                 # Trained model weights
├── brain_tumor_model.png     # Model architecture diagram 
├── training2.ipynb           # Jupyter notebook with training and visualization code 
├── training_log_brain.csv    # CSV log with training metrics per epoch 
├── README.md                 # Project
```

## Installation

1. **Clone the repository**:
```
   git clone https://github.com/yourusername/brain-tumor-detection.git
   cd brain-tumor-detection
```
2. **Install required packages**
 ```
   pip install -r requirements.txt
 ```

## Usage

1. **Load the Model:** Use ```brain.pth``` to load the trained model weights in your inference script or notebook.

2. **Run Inference:** Use the code in ```training2.ipynb``` or another script to classify new MRI images and visualize important regions with Grad-CAM.
```
   from model import load_model
   model = load_model('brain.pth')
```

3. **View Training Logs:** Open ```training_log_brain.csv``` to review training and validation loss, accuracy, and other metrics across epochs.

## Results
The model achieves high classification accuracy on the validation and test sets across all four classes. Grad-CAM visualizations further help in interpreting the model's predictions, showing heatmaps on image regions with the highest influence on the model's decisions.

## References
- **Dataset:** Brain Tumor MRI Dataset
- **Libraries:** PyTorch, torchvision, OpenCV, Grad-CAM
