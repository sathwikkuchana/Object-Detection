# Object-Detection

This project implements a Convolutional Neural Network (CNN) from scratch to perform image classification and object localization tasks. It uses the MNIST dataset to synthesize a custom dataset where each "digit" image is placed on a black canvas of 75 x 75 dimensions at random locations. The model is trained to classify the main subject in an image (digit) and localize it by drawing bounding boxes around it.

# Table of Contents
1.Introduction

2.Dataset Synthesis

3.Visualization Utilities

4.Network Architecture

5.Training and Validation

6.Intersection over Union Metric

7.Visualize Predictions

8.Usage

9.Dependencies

# Introduction
The CNN architecture includes a feature extractor, classifier, and bounding box regression layers. The feature extractor comprises convolutional and pooling layers to extract image features. The classifier predicts the digit category (0-9), and the bounding box regression predicts the coordinates of the bounding box (xmin, ymin, xmax, ymax) around the digit.

# Dataset Synthesis
Each digit image is placed randomly on a 75x75 black canvas.
Bounding box coordinates are calculated for each digit image.
![image](https://github.com/sathwikkuchana/Object-Detection/assets/37955149/3c04ff07-eac3-43bd-b9e9-6793f2215bf7)

# Visualization Utilities
Utilities are provided to visualize the data and predictions:
draw_bounding_boxes_on_image_array: Draws bounding boxes around digits on an image array.
draw_bounding_boxes_on_image: Draws bounding boxes on an image.
display_digits_with_boxes: Displays digits with their true and predicted bounding boxes.

![image](https://github.com/sathwikkuchana/Object-Detection/assets/37955149/973480f1-da34-4a4d-99ef-128b6c4a91f4)
![image](https://github.com/sathwikkuchana/Object-Detection/assets/37955149/62519043-db2c-48b7-9668-284967409b4d)
# Network Architecture
The CNN architecture includes:

Feature Extractor: Convolutional layers for feature extraction.
Dense Layers: Flatten and dense layers for further processing.
Classifier: Predicts the digit category (0-9).
Bounding Box Regression: Predicts bounding box coordinates.

![image](https://github.com/sathwikkuchana/Object-Detection/assets/37955149/a5585bc7-5a2d-4340-bcc1-b450ed3b2782)

# Training and Validation
The model is trained using the training dataset and validated using the validation dataset.

Optimizer: Adam optimizer.
Loss Functions: Categorical cross-entropy for classification, Mean Squared Error (MSE) for bounding box regression.
Metrics: Accuracy for classification, MSE for bounding box prediction.
![image](https://github.com/sathwikkuchana/Object-Detection/assets/37955149/56c2d2b6-f2c1-49d0-9622-beb1fa8b4fd7)

# Intersection over Union Metric
The Intersection over Union (IoU) metric is calculated to evaluate the model's performance in bounding box prediction.

# Visualize Predictions
The final step visualizes predictions by overlaying true and predicted bounding boxes on digit images. Green boxes represent true labels, and red boxes represent predicted bounding boxes.
![image](https://github.com/sathwikkuchana/Object-Detection/assets/37955149/bb57f43d-d6c6-4396-bef4-cb0759b1d92f)

# Usage
Clone the repository.
Ensure dependencies are installed (requirements.txt).
Run the code using Python (python main.py).
# Dependencies
TensorFlow
NumPy
Matplotlib
PIL (Pillow)
TensorFlow Datasets (tfds)
