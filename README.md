# Capstone_Medical_Image_Diagnosis_Using_Deep_Learning

Project Overview:
This project focuses on the automatic diagnosis of pneumonia from chest X-ray images using Deep Learning techniques. The objective is to assist healthcare professionals by providing a computer-aided diagnostic system capable of classifying chest X-ray images as NORMAL or PNEUMONIA.

Objectives:
Develop an automated system for pneumonia detection from chest X-ray images.
Reduce diagnostic workload for healthcare professionals.
Compare the performance of multiple deep learning architectures.
Evaluate model effectiveness using standard classification metrics.

Dataset: Chest X-Ray Images (Pneumonia)

Classes
  NORMAL
  PNEUMONIA
Dataset Structure
Dataset/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── val/
    ├── NORMAL/
    └── PNEUMONIA/


Technologies Used:
  Python
  TensorFlow
  Keras
  NumPy
  Pandas
  Matplotlib
  Seaborn
  OpenCV
  Scikit-learn
  Jupyter Notebook

Data Preprocessing:
The following preprocessing techniques were applied:
  Image resizing
  Pixel normalization
  Data augmentation
  Batch generation using ImageDataGenerator
  Train, Validation, and Test split

Deep Learning Models:
1. Convolutional Neural Network (CNN)
A custom CNN architecture was developed using:
  Convolution Layers
  Max Pooling Layers
  Dropout Layers
  Dense Layers
  Softmax/Sigmoid Output Layer
2. MobileNet
Transfer Learning was applied using MobileNet to leverage pre-trained ImageNet weights for improved feature extraction and classification performance.

 
Evaluation Metrics:
The models were evaluated using:
  Accuracy
  Precision
  Recall
  F1-Score
  Confusion Matrix
  Classification Report

Results:
  Model	                Accuracy
0	Custom CNN	            82.53
1	MobileNetV2	            88.30
2	Improved MobileNetV2	  87.82

Sample Prediction:
The trained model can predict whether a chest X-ray image belongs to:
  NORMAL
  PNEUMONIA

Outputs:
  <img width="940" height="490" alt="image" src="https://github.com/user-attachments/assets/14686e4c-e636-4f19-a7c6-03e995c4456b" />
  <img width="940" height="485" alt="image" src="https://github.com/user-attachments/assets/cd2a5d4e-14c5-4456-a8cb-202347e04e9d" />
  <img width="940" height="496" alt="image" src="https://github.com/user-attachments/assets/dfe3eb88-a397-4634-9161-137f9810f689" />


Future Enhancements:
  Multi-class disease classification
  Integration with hospital systems
  Explainable AI (XAI) techniques
  Real-time clinical deployment
  Ensemble deep learning models
  
