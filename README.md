# Machine_Learning_Exercises

# Learning from Data - Homework Solutions

This repository contains solutions for homework assignments based on the textbook "Learning from Data." The assignments focus on implementing and analyzing various machine learning algorithms and concepts. Each exercise corresponds to specific problems from the book and is implemented in Python. These assignments are part of the Machine Learning course CMPE-257 at San Jose State University (SJSU).

## File Descriptions

### 1. `exercise_1.py`
This script implements tasks related to fundamental machine learning concepts, including:
- **Linear Perceptron Tasks:**
  - Misclassification detection using dot product operations.
  - Solving traditional linear equations.
- **Data Preprocessing and Visualization:**
  - Preprocessing digit classification data (digits 1 and 5).
  - Extracting intensity and symmetry features.
  - Generating scatter plots to visualize data features.

### 2. `exercise_2.py`
This script focuses on algorithmic implementations and data simulations:
- **Gradient Computation:**
  - Numerical gradient calculation for error functions.
- **Perceptron Learning Algorithm (PLA):**
  - Implemented on double semi-circle datasets.
- **Linear Regression and Pocket Algorithm:**
  - Linear regression for hypothesis estimation.
  - Pocket algorithm for binary classification tasks.
- **Data Simulation:**
  - Custom data generation for specific geometrical configurations.
- **Performance Comparison:**
  - Comparing pocket algorithm and linear regression results on various datasets.

### 3. `exercise_3.py`
This script explores neural network implementations and training processes:
- **Neural Network Implementation:**
  - Forward and backward propagation for single-hidden-layer neural networks.
  - Training using gradient descent.
- **Feature Engineering:**
  - Extracting intensity and symmetry features from image data.
  - Augmenting features with polynomial transformations.
- **Performance Evaluation:**
  - Training and testing a neural network on digit classification tasks.
  - Reporting in-sample and test errors.
- **Mini-Batch Gradient Descent:**
  - Improved training with mini-batch updates.

### 4. `exercise_4.py`
This script focuses on Support Vector Machines (SVM):
- **SVM Classifier:**
  - Linear, polynomial, and RBF kernels.
  - Hyperparameter tuning using GridSearchCV.
  - Comparison of training and validation accuracies.
- **Visualization:**
  - Heatmap visualization of cross-validation accuracies for different hyperparameters.
- **Model Comparison:**
  - Bar chart comparing test accuracies across multiple models, including PLA, Pocket Algorithm, Neural Networks, and SVMs.

## Dataset
The dataset used in these assignments is derived from the `ZipDigits.train` and `ZipDigits.test` files, consisting of digit images (digits 1 and 5) with pixel values. Features such as intensity and symmetry are computed to represent each image in a reduced feature space.

## How to Run
1. Ensure you have Python installed with the following libraries:
   - `numpy`
   - `matplotlib`
   - `scikit-learn`
   - `pandas`
2. Clone the repository and navigate to the directory.
3. Run the scripts using:
   ```bash
   python exercise_X.py
   ```
   Replace `X` with the exercise number (1-4).

## Learning Objectives
- Understand fundamental concepts in machine learning such as linear classification, feature extraction, and model evaluation.
- Gain hands-on experience with algorithms like PLA, pocket algorithm, and SVMs.
- Explore advanced topics like neural networks and mini-batch gradient descent.

## Acknowledgments
The problems solved in this repository are inspired by and based on the textbook "Learning from Data." These assignments were completed as part of the Machine Learning course CMPE-257 at San Jose State University (SJSU).

Feel free to contribute by suggesting improvements or reporting issues!

