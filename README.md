# Learning from Data - Homework Solutions and Additional Implementations

This repository contains solutions for homework assignments based on the textbook "Learning from Data," as well as additional implementations of machine learning algorithms. The assignments focus on implementing and analyzing various machine learning algorithms and concepts. Each exercise corresponds to specific problems from the book and is implemented in Python. These assignments were completed as part of the Machine Learning course CMPE-257 at San Jose State University (SJSU).

## File Descriptions

### 1. `exercise_1.py`
This script implements tasks related to fundamental machine learning concepts, including:

- **Linear Perceptron Tasks**:
  - Misclassification detection using dot product operations.
  - Solving traditional linear equations.
- **Data Preprocessing and Visualization**:
  - Preprocessing digit classification data (digits 1 and 5).
  - Extracting intensity and symmetry features.
  - Generating scatter plots to visualize data features.

### 2. `exercise_2.py`
This script focuses on algorithmic implementations and data simulations:

- **Gradient Computation**:
  - Numerical gradient calculation for error functions.
- **Perceptron Learning Algorithm (PLA)**:
  - Implemented on double semi-circle datasets.
- **Linear Regression and Pocket Algorithm**:
  - Linear regression for hypothesis estimation.
  - Pocket algorithm for binary classification tasks.
- **Data Simulation**:
  - Custom data generation for specific geometrical configurations.
- **Performance Comparison**:
  - Comparing pocket algorithm and linear regression results on various datasets.

### 3. `exercise_3.py`
This script explores neural network implementations and training processes:

- **Neural Network Implementation**:
  - Forward and backward propagation for single-hidden-layer neural networks.
  - Training using gradient descent.
- **Feature Engineering**:
  - Extracting intensity and symmetry features from image data.
  - Augmenting features with polynomial transformations.
- **Performance Evaluation**:
  - Training and testing a neural network on digit classification tasks.
  - Reporting in-sample and test errors.
- **Mini-Batch Gradient Descent**:
  - Improved training with mini-batch updates.

### 4. `exercise_4.py`
This script focuses on Support Vector Machines (SVM):

- **SVM Classifier**:
  - Linear, polynomial, and RBF kernels.
  - Hyperparameter tuning using GridSearchCV.
- **Visualization**:
  - Heatmap visualization of cross-validation accuracies for different hyperparameters.
- **Model Comparison**:
  - Bar chart comparing test accuracies across multiple models, including PLA, Pocket Algorithm, Neural Networks, and SVMs.

### 5. `pla_and_pocket.py`
This script demonstrates the implementation of the Perceptron Learning Algorithm (PLA) and the Pocket algorithm:

- **PLA**:
  - Updates weights iteratively to minimize misclassification errors.
  - Visualization of the decision boundary on a synthetic dataset.
- **Pocket Algorithm**:
  - Maintains the best-performing weights during training.
  - Visualizes the pocket decision boundary.

### 6. `knn_based_recommendation_system.py`
This script implements a content-based movie recommendation system using the k-Nearest Neighbors (k-NN) algorithm:

- **Data Loading and Preprocessing**:
  - Utilizes movie and rating datasets to compute movie rating counts.
  - Filters movies based on a popularity threshold.
- **Matrix Construction**:
  - Builds a pivot table for user-movie ratings.
- **k-NN Implementation**:
  - Trains a k-NN model to recommend similar movies based on user ratings.
  - Displays recommendations and their distances.

### 7. `linear_regression.py`
This script demonstrates linear regression to estimate a target function:

- **Data Generation**:
  - Creates synthetic data based on a true target function.
- **Visualization**:
  - Plots the true target function and the linear regression hypothesis.
- **Linear Regression Implementation**:
  - Computes the regression weights using matrix operations.
  - Displays the estimated weights and hypothesis line.

## Dataset
The primary datasets include:
- **Digit Classification**: ZipDigits.train and ZipDigits.test files containing pixel values for digit images.
- **Synthetic Datasets**: Custom-generated data for PLA, Pocket, and regression tasks.
- **Movie Recommendation**: Movies and ratings datasets for k-NN-based recommendations.

## How to Run
Ensure you have Python installed with the following libraries:
- `numpy`
- `matplotlib`
- `scikit-learn`
- `pandas`
- `scipy`

Clone the repository and navigate to the directory. Run the scripts using:
```bash
python <script_name>.py
```
Replace `<script_name>` with the desired file name (e.g., `exercise_1.py`).

## Learning Objectives
- Understand and implement machine learning algorithms such as PLA, pocket algorithm, k-NN, SVMs, and neural networks.
- Gain hands-on experience with feature extraction, data preprocessing, and performance evaluation.
- Explore advanced topics such as mini-batch gradient descent and content-based recommendation systems.

## Acknowledgments
The problems solved in this repository are inspired by and based on the textbook "Learning from Data." These assignments were completed as part of the Machine Learning course CMPE-257 at San Jose State University (SJSU).

Feel free to contribute by suggesting improvements or reporting issues!

