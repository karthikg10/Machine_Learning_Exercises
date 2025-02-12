# -*- coding: utf-8 -*-
"""ML_Homework-2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1O1vRFWfvslGsIp19i8jcd5jyMqiaXnN8
"""

# Task-1 HP2
import numpy as np

def er_func(w):
    return (w[0] - 2) ** 2 + (w[1] + 3) ** 2

w = np.array([1.0, -2.0])


def grad(error_func, w):
    h = 1e-5
    gradient = np.zeros_like(w)

    for i in range(len(w)):

        w_perturbed = w.copy()
        w_perturbed[i] += h


        gradient[i] = (error_func(w_perturbed) - error_func(w)) / h

    return gradient

# Compute the gradient
gradient = grad(er_func, w)
print("Gradient:", gradient)

# Task-2:LP1

import numpy as np
import matplotlib.pyplot as plt

#code snippet from "hw-2_3.1"
# Parameters
rad = 10
thk = 5
sep = 5

# n data points, (x1, y1) are the coordinates of the top semi-circle
def generate_data(rad, thk, sep, n, x1=0, y1=0):
    # center of the top semi-circle
    X1 = x1
    Y1 = y1

    # center of the bottom semi-circle
    X2 = X1 + rad + thk / 2
    Y2 = Y1 - sep

    # data points in the top semi-circle
    top = []
    # data points in the bottom semi-circle
    bottom = []

    # parameters
    r1 = rad + thk
    r2 = rad

    cnt = 1
    while cnt <= n:
        # uniformly generated points
        x = np.random.uniform(-r1, r1)
        y = np.random.uniform(-r1, r1)

        d = x**2 + y**2
        if d >= r2**2 and d <= r1**2:
            if y > 0:
                top.append([X1 + x, Y1 + y])
                cnt += 1
            else:
                bottom.append([X2 + x, Y2 + y])
                cnt += 1

    return top, bottom

# Define the PLA algorithm
def perceptron(X, Y):
    w = np.zeros(X.shape[1])
    converged = False
    iters = 0
    max_iters = 10000
    lr = 0.1

    while not converged and iters < max_iters:
        misclassified = []
        for i in range(len(X)):
            if np.sign(np.dot(w, X[i])) != Y[i]:
                misclassified.append(i)

        if not misclassified:
            converged = True
        else:
            random_index = np.random.choice(misclassified)
            w += lr * Y[random_index] * X[random_index]
            iters += 1

    return w, iters

# Generate data for the double semi-circle task
top, bottom = generate_data(rad, thk, sep, 1000)

# Combine data points from both classes
data = top + bottom

# Extract X and Y coordinates for plotting
X1 = [i[0] for i in top]
Y1 = [i[1] for i in top]

X2 = [i[0] for i in bottom]
Y2 = [i[1] for i in bottom]

# Assign labels (-1) for the bottom semi-circle and (+1) for the top semi-circle
X = np.array(data)
Y = np.array([-1] * len(top) + [1] * len(bottom))

# Train the PLA on the generated data
w, iters = perceptron(X, Y)

# Scatter plot of the generated data
plt.scatter(X1, Y1, c='b', s=1, label='Class +1 (Top Semi-Circle)')
plt.scatter(X2, Y2, c='r', s=1, label='Class -1 (Bottom Semi-Circle)')

# Plot the decision boundary
if len(w) == 3:
    x_decision = np.linspace(-30, 30, 100)
    y_decision = (-w[0] - w[1] * x_decision) / w[2]
    plt.plot(x_decision, y_decision, 'g-', label='Decision Boundary')
else:
    x_decision = np.linspace(-30, 30, 100)
    y_decision = (-w[0] * x_decision) / w[1]
    plt.plot(x_decision, y_decision, 'g-', label='Decision Boundary')

plt.title('PLA for Double Semi-Circle Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

print(f'PLA converged in {iters} iterations')

import numpy as np
import matplotlib.pyplot as plt
import random
# code snippet from the class-acticity
# Generates N random points for linear regression on a line w
def generate_lr(N=20, w0=0, w1=1):
    n = 0
    X1 = []
    Y = []
    while n < N:
        x1 = random.uniform(-10, 10)
        x2 = random.uniform(-1, 1)
        y = w0 + w1 * x1 + x2
        X1.append(x1)
        Y.append(y)
        n += 1
    data = [np.array([1, X1[i], Y[i]]) for i in range(N)]
    return data

# True target function parameters
f_w0, f_w1 = 1, 1
N = 50
data = generate_lr(N, f_w0, f_w1)

# A function to plot the target function
def abline(slope, intercept, label_text):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', label=label_text)

X1 = [i[1] for i in data]
Y = [i[2] for i in data]

# Scatter plot of data points
plt.scatter(X1, Y, s=10, label='Data Points')

# Plot for true target function
abline(f_w1, f_w0, 'True Target Function')

# Linear Regression
X = [[i[0], i[1]] for i in data]
Y = [i[2] for i in data]
xTx = np.matmul(np.transpose(X), X)
xTx_inv = np.linalg.inv(xTx)
X_pi = np.matmul(xTx_inv, np.transpose(X))
w = np.matmul(X_pi, Y)

# Plot for linear regression hypothesis
abline(w[1], w[0], 'Linear Regression Hypothesis')

plt.xlabel('X1')
plt.ylabel('Y')
plt.legend()
plt.title('Linear Regression for Target Function Estimation')
plt.grid(True)

plt.show()

print('Estimated Weights (intercept, slope):', w)

# Task-2: LP2

import numpy as np
import random

# Parameters
rad = 10
thk = 5
sep = -5  # Set sep to -5
n = 2000  # Generate 2000 examples

# n data points, (x1, y1) are the coordinates of the top semi-circle
def generate_data(rad, thk, sep, n, x1=0, y1=0):
    # center of the top semi-circle
    X1 = x1
    Y1 = y1

    # center of the bottom semi-circle
    X2 = X1 + rad + thk / 2
    Y2 = Y1 - sep

    # data points in the top semi-circle
    top = []
    # data points in the bottom semi-circle
    bottom = []

    # parameters
    r1 = rad + thk
    r2 = rad

    cnt = 1
    while cnt <= n:
        # uniformly generated points
        x = np.random.uniform(-r1, r1)
        y = np.random.uniform(-r1, r1)

        d = x**2 + y**2
        if d >= r2**2 and d <= r1**2:
            if y > 0:
                top.append([X1 + x, Y1 + y])
                cnt += 1
            else:
                bottom.append([X2 + x, Y2 + y])
                cnt += 1

    return top, bottom

# Generate data for the double semi-circle task
top, bottom = generate_data(rad, thk, sep, n)

import numpy as np
import random
import matplotlib.pyplot as plt

# Parameters for the Pocket Algorithm
max_iterations_pocket = 100000
initial_learning_rate = 0.1  # Initial learning rate


def pocket_algorithm(data, max_iterations, initial_learning_rate):
    w = np.zeros(len(data[0]) - 1)
    pocket_w = w.copy()
    min_error = len(data)
    error_history = []

    for t in range(max_iterations):
        misclassified = []

        for point in data:
            x = point[:-1]
            y = point[-1]

            if np.sign(np.dot(w, x)) != np.sign(y):
                misclassified.append(point)

        error = len(misclassified)
        error_history.append(error)

        if error < min_error:
            min_error = error
            pocket_w = w.copy()

        if not misclassified:
            break
        else:
            misclassified_point = random.choice(misclassified)
            misclassified_point = np.array(misclassified_point)
            learning_rate = initial_learning_rate / (1 + t)
            w += learning_rate * misclassified_point[-1] * misclassified_point[:-1]

    return pocket_w, error_history

# Run the Pocket Algorithm on the generated data
data = top + bottom
data_with_labels = [[*point, 1] for point in top] + [[*point, -1] for point in bottom]

pocket_w, error_history = pocket_algorithm(data_with_labels, max_iterations_pocket, initial_learning_rate)

# Plot Ein (misclassified points) vs. iteration number t
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(error_history) + 1), error_history)
plt.title('Pocket Algorithm: Ein vs. Iteration Number')
plt.xlabel('Iteration Number (t)')
plt.ylabel('Ein (Misclassified Points)')
plt.grid(True)
plt.show()

print(f'Pocket Algorithm converged in {len(error_history)} iterations')

import numpy as np
import random
import matplotlib.pyplot as plt


max_iterations_pocket = 100000
initial_learning_rate = 0.1


def pocket_algorithm(data, max_iterations, initial_learning_rate):
    w = np.zeros(len(data[0]) - 1)
    pocket_w = w.copy()
    min_error = len(data)
    error_history = []

    for t in range(max_iterations):
        misclassified = []

        for point in data:
            x = point[:-1]
            y = point[-1]

            if np.sign(np.dot(w, x)) != np.sign(y):
                misclassified.append(point)

        error = len(misclassified)
        error_history.append(error)

        if error < min_error:
            min_error = error
            pocket_w = w.copy()

        if not misclassified:
            break
        else:
            misclassified_point = random.choice(misclassified)
            misclassified_point = np.array(misclassified_point)
            learning_rate = initial_learning_rate / (1 + t)
            w += learning_rate * misclassified_point[-1] * misclassified_point[:-1]

    return pocket_w, error_history

# Run the Pocket Algorithm on the generated data
data = top + bottom
data_with_labels = [[*point, 1] for point in top] + [[*point, -1] for point in bottom]

pocket_w, error_history = pocket_algorithm(data_with_labels, max_iterations_pocket, initial_learning_rate)


# Plot the data and the final hypothesis
plt.figure(figsize=(10, 6))

# Data points from class +1
X1 = [point[0] for point in top]
Y1 = [point[1] for point in top]
plt.scatter(X1, Y1, c='b', s=1, label='Class +1 (Top Semi-Circle)')

# Data points from class -1
X2 = [point[0] for point in bottom]
Y2 = [point[1] for point in bottom]
plt.scatter(X2, Y2, c='r', s=1, label='Class -1 (Bottom Semi-Circle)')

if len(pocket_w) == 2:
    x_decision = np.linspace(-25, 25, 100)
    y_decision = (-pocket_w[0] - pocket_w[1] * x_decision)
    plt.plot(x_decision, y_decision, 'g-', label='Decision Boundary')
else:

    print("Handling higher-dimensional weight vector is not implemented in this example.")


plt.title('Pocket Algorithm: Data and Final Hypothesis')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

print(f'Pocket Algorithm converged in {len(error_history)} iterations')

#Task2- HP, d


import numpy as np
import random
import time

# Function to generate the double semi-circle data
def generate_double_semi_circle(rad, thk, sep, num_points):
    data = []
    for _ in range(num_points):
        r = random.uniform(rad, rad + thk)
        angle = random.uniform(0, np.pi)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        if x <= 0:
            x += sep / 2
        else:
            x -= sep / 2
        data.append((x, y))
    return data

# Function to implement the Pocket Algorithm
def perceptron_learning_algorithm(data, max_iterations):
    w = np.zeros(3)
    pocket_w = np.zeros(3)
    min_error = len(data)

    for _ in range(max_iterations):
        misclassified = []
        for point in data:
            x = np.array([1, point[0], point[1]])
            if np.sign(np.dot(w, x)) != 1:
                misclassified.append(x)

        error = len(misclassified)

        if error < min_error:
            min_error = error
            pocket_w = w.copy()

        if not misclassified:
            break
        else:
            misclassified_point = random.choice(misclassified)
            w += misclassified_point

    return pocket_w, min_error

# Function to implement linear regression for classification
def linear_regression_classification(data):
    X = np.array(data)
    Y = np.array([-1] * len(data))
    Y[len(data) // 2:] = 1

    X = np.hstack((np.ones((len(data), 1)), X))
    X_T = np.transpose(X)
    w = np.linalg.inv(X_T @ X) @ X_T @ Y

    return w

# Constants for the problem
rad = 10
thk = 5
sep = -5
num_points_per_class = 1000
total_points = 2 * num_points_per_class
max_iterations_pocket = 100000

# Generate the double semi-circle data
data_class1 = generate_double_semi_circle(rad, thk, sep, num_points_per_class)
data_class2 = generate_double_semi_circle(rad, thk, -sep, num_points_per_class)

# Combine the data points from both classes
data = data_class1 + data_class2

# Measure the computation time and quality of the solution for Pocket Algorithm
start_time_pocket = time.time()
pocket_w, min_error = perceptron_learning_algorithm(data, max_iterations_pocket)
end_time_pocket = time.time()
pocket_computation_time = end_time_pocket - start_time_pocket

# Measure the computation time and quality of the solution for Linear Regression
start_time_lr = time.time()
lr_w = linear_regression_classification(data)
end_time_lr = time.time()
lr_computation_time = end_time_lr - start_time_lr

# Evaluate quality of solutions (classification accuracy)
def classify_with_weights(weights, point):
    return np.sign(np.dot(weights, np.array([1, point[0], point[1]])))

pocket_accuracy = sum([classify_with_weights(pocket_w, point) for point in data]) / total_points
lr_accuracy = sum([classify_with_weights(lr_w, point) for point in data]) / total_points

# Print and compare results
print(f'Pocket Algorithm Computation Time: {pocket_computation_time:.6f} seconds')
print(f'Linear Regression Computation Time: {lr_computation_time:.6f} seconds')
print(f'Pocket Algorithm Minimum Ein (Misclassified Points): {min_error}')
print(f'Pocket Algorithm Classification Accuracy: {pocket_accuracy:.4f}')
print(f'Linear Regression Classification Accuracy: {lr_accuracy:.4f}')

# Task-2 HP:e
import numpy as np
import random
import matplotlib.pyplot as plt


max_iterations_pocket = 100000
initial_learning_rate = 0.1


def pocket_algorithm(data, max_iterations, initial_learning_rate):
    w = np.zeros(len(data[0]) - 1)
    pocket_w = w.copy()
    min_error = len(data)
    error_history = []

    for t in range(max_iterations):
        misclassified = []

        for point in data:
            x = point[:-1]
            y = point[-1]

            if np.sign(np.dot(w, x)) != np.sign(y):
                misclassified.append(point)

        error = len(misclassified)
        error_history.append(error)

        if error < min_error:
            min_error = error
            pocket_w = w.copy()

        if not misclassified:
            break
        else:
            misclassified_point = random.choice(misclassified)
            learning_rate = initial_learning_rate / (1 + t)
            w += learning_rate * misclassified_point[-1] * np.array(misclassified_point[:-1])

    return pocket_w, error_history

# Run the Pocket Algorithm on the generated data with a 3rd order polynomial feature transform
data_with_labels = [[*point, 1] for point in top] + [[*point, -1] for point in bottom]

pocket_w, error_history = pocket_algorithm(data_with_labels, max_iterations_pocket, initial_learning_rate)

# Plot Ein (misclassified points) vs. iteration number t
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(error_history) + 1), error_history)
plt.title('Pocket Algorithm: Ein vs. Iteration Number')
plt.xlabel('Iteration Number (t)')
plt.ylabel('Ein (Misclassified Points)')
plt.grid(True)
plt.show()

print(f'Pocket Algorithm converged in {len(error_history)} iterations')

# Plot the data points and the final hypothesis

# Task-3 LP1

import numpy as np
import pandas as pd

# Load training and test datasets
train_data = pd.read_csv('ZipDigits.train', header=None, delimiter=' ')
test_data = pd.read_csv('ZipDigits.test', header=None, delimiter=' ')

# Separate labels and pixel values
train_labels = train_data.iloc[:, 0].values
train_pixels = train_data.iloc[:, 1:].values

test_labels = test_data.iloc[:, 0].values
test_pixels = test_data.iloc[:, 1:].values

# Filter the dataset for digits '1' and '5'
selected_train_indices = np.where((train_labels == 1) | (train_labels == 5))
selected_test_indices = np.where((test_labels == 1) | (test_labels == 5))

# Select only the relevant samples
train_labels = train_labels[selected_train_indices]
train_pixels = train_pixels[selected_train_indices]
test_labels = test_labels[selected_test_indices]
test_pixels = test_pixels[selected_test_indices]

# Convert labels to binary classification: '1' and '-1'
train_labels[train_labels == 1] = 1
train_labels[train_labels == 5] = -1
test_labels[test_labels == 1] = 1
test_labels[test_labels == 5] = -1

# Load the data
data = np.loadtxt('ZipDigits.train')

# Filter for digits '1' and '5' and convert labels
filtered_data = data[(data[:, 0] == 1) | (data[:, 0] == 5)]
filtered_data[:, 0] = np.where(filtered_data[:, 0] == 1, 1, -1)

def intensity_feature(data):

    return np.sum(data[1:])

def symmetry_feature(data):

    left_side = data[1:128]
    right_side = data[129:256]
    return np.mean(left_side - right_side)

def perceptron_train(data, max_iterations=1000):
    num_samples, num_features = data.shape
    weights = np.zeros(num_features - 1)
    for iteration in range(max_iterations):
        misclassified = 0
        for sample in data:
            label, features = sample[0], sample[1:]
            prediction = np.dot(weights, features)
            if label * prediction <= 0:
                misclassified += 1
                weights += label * features
        if misclassified == 0:
            break
    return weights

# Train the PLA on the filtered training dataset
final_weights_train = perceptron_train(filtered_data)


plt.figure(figsize=(8, 6))
plt.scatter(filtered_data[:, 1], filtered_data[:, 2], c=filtered_data[:, 0], s=20)
x_train = np.linspace(-1, 1, 100)
if abs(final_weights_train[2]) > 1e-6:
    y_train = (-final_weights_train[1] * x_train - final_weights_train[0]) / final_weights_train[2]
else:
    y_train = np.zeros_like(x_train)
plt.plot(x_train, y_train, '-r', label='Hypothesis Line (Training Data)')
plt.xlabel('Intensity Feature')
plt.ylabel('Symmetry Feature')
plt.legend()
plt.title('Training Data')
plt.show()

def pocket_algorithm(data, max_updates=1000):
    num_samples, num_features = data.shape
    weights = np.zeros(num_features - 1)
    best_weights = weights.copy()
    best_misclassified = num_samples

    for _ in range(max_updates):
        misclassified = 0
        for sample in data:
            label, features = sample[0], sample[1:]
            prediction = np.dot(weights, features)
            if label * prediction <= 0:
                misclassified += 1
                weights += label * features


        if misclassified < best_misclassified:
            best_weights = weights.copy()
            best_misclassified = misclassified

    return best_weights

final_weights = pocket_algorithm(filtered_data, max_updates=1000)

import matplotlib.pyplot as plt


plt.scatter(filtered_data[:, 1], filtered_data[:, 2], c=filtered_data[:, 0], s=20)

x = np.linspace(-1, 1, 100)
if abs(final_weights[2]) > 1e-6:
    y = (-final_weights[1] * x - final_weights[0]) / final_weights[2]
else:
    y = np.zeros_like(x)
plt.plot(x, y, '-g', label='Hypothesis Line')

plt.xlabel('Intensity Feature')
plt.ylabel('Symmetry Feature')
plt.legend()
plt.show()

import numpy as np

# Load and preprocess the training dataset
training_data = np.loadtxt('ZipDigits.train')
training_filtered = training_data[(training_data[:, 0] == 1) | (training_data[:, 0] == 5)]
training_labels = np.where(training_filtered[:, 0] == 1, 1, -1)
training_features = np.column_stack((np.ones(len(training_filtered)), training_filtered[:, 1], training_filtered[:, 2],
    training_filtered[:, 1] ** 3, training_filtered[:, 2] ** 3,
    (training_filtered[:, 1] ** 2) * training_filtered[:, 2],
    training_filtered[:, 1] * (training_filtered[:, 2] ** 2)))

# Load and preprocess the test dataset
test_data = np.loadtxt('ZipDigits.test')
test_filtered = test_data[(test_data[:, 0] == 1) | (test_data[:, 0] == 5)]
test_labels = np.where(test_filtered[:, 0] == 1, 1, -1)
test_features = np.column_stack((np.ones(len(test_filtered)), test_filtered[:, 1], test_filtered[:, 2],
    test_filtered[:, 1] ** 3, test_filtered[:, 2] ** 3,
    (test_filtered[:, 1] ** 2) * test_filtered[:, 2],
    test_filtered[:, 1] * (test_filtered[:, 2] ** 2)))

def pocket_algorithm(data, labels, max_updates=1000):
    num_samples, num_features = data.shape
    weights = np.zeros(num_features)
    best_weights = weights.copy()
    best_error = num_samples

    for _ in range(max_updates):
        misclassified = 0
        for i in range(num_samples):
            prediction = np.sign(np.dot(data[i], weights))
            if prediction != labels[i]:
                misclassified += 1
                weights += labels[i] * data[i]

        test_error = (num_samples - misclassified) / num_samples


        if test_error < best_error:
            best_weights = weights.copy()
            best_error = test_error

    return best_weights, best_error

final_weights, test_error = pocket_algorithm(training_features, training_labels, max_updates=1000)

print(f'Test Error: {100 * test_error:.2f}%')

import numpy as np

# Load and preprocess the training dataset
training_data = np.loadtxt('ZipDigits.train')
training_filtered = training_data[(training_data[:, 0] == 1) | (training_data[:, 0] == 5)]
training_labels = np.where(training_filtered[:, 0] == 1, 1, -1)
training_features = np.column_stack((np.ones(len(training_filtered)), training_filtered[:, 1], training_filtered[:, 2],
    training_filtered[:, 1] ** 3, training_filtered[:, 2] ** 3,
    (training_filtered[:, 1] ** 2) * training_filtered[:, 2],
    training_filtered[:, 1] * (training_filtered[:, 2] ** 2)))

# Load and preprocess the test dataset
test_data = np.loadtxt('ZipDigits.test')
test_filtered = test_data[(test_data[:, 0] == 1) | (test_data[:, 0] == 5)]
test_labels = np.where(test_filtered[:, 0] == 1, 1, -1)
test_features = np.column_stack((np.ones(len(test_filtered)), test_filtered[:, 1], test_filtered[:, 2],
    test_filtered[:, 1] ** 3, test_filtered[:, 2] ** 3,
    (test_filtered[:, 1] ** 2) * test_filtered[:, 2],
    test_filtered[:, 1] * (test_filtered[:, 2] ** 2)))

def perceptron_algorithm(data, labels, max_iterations=1000):
    num_samples, num_features = data.shape
    weights = np.zeros(num_features)

    for _ in range(max_iterations):
        misclassified = 0
        for i in range(num_samples):
            prediction = np.sign(np.dot(data[i], weights))
            if prediction != labels[i]:
                misclassified += 1
                weights += labels[i] * data[i]

        if misclassified == 0:
            break

    return weights
final_weights = perceptron_algorithm(training_features, training_labels, max_iterations=1000)

def test_perceptron(data, labels, weights):
    num_samples = len(data)
    misclassified = 0
    for i in range(num_samples):
        prediction = np.sign(np.dot(data[i], weights))
        if prediction != labels[i]:
            misclassified += 1
    test_error = misclassified / num_samples
    return test_error

test_error = test_perceptron(test_features, test_labels, final_weights)
print(f'Test Error: {100 * test_error:.2f}%')

# LP-2
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the training dataset
training_data = np.loadtxt('ZipDigits.train')
training_filtered = training_data[(training_data[:, 0] == 1) | (training_data[:, 0] == 5)]
training_labels = np.where(training_filtered[:, 0] == 1, 1, -1)
training_features = np.column_stack((np.ones(len(training_filtered)), training_filtered[:, 1], training_filtered[:, 2], training_filtered[:, 1] ** 3, training_filtered[:, 2] ** 3))

# Load and preprocess the test dataset
test_data = np.loadtxt('ZipDigits.test')
test_filtered = test_data[(test_data[:, 0] == 1) | (test_data[:, 0] == 5)]
test_labels = np.where(test_filtered[:, 0] == 1, 1, -1)
test_features = np.column_stack((np.ones(len(test_filtered)), test_filtered[:, 1], test_filtered[:, 2], test_filtered[:, 1] ** 3, test_filtered[:, 2] ** 3))

def pocket_algorithm(data, labels, max_updates=1000):
    num_samples, num_features = data.shape
    weights = np.zeros(num_features)
    best_weights = weights.copy()
    best_error = num_samples

    for _ in range(max_updates):
        misclassified = 0
        for i in range(num_samples):
            prediction = np.sign(np.dot(data[i], weights))
            if prediction != labels[i]:
                misclassified += 1
                weights += labels[i] * data[i]


        test_error = (num_samples - misclassified) / num_samples


        if test_error < best_error:
            best_weights = weights.copy()
            best_error = test_error

    return best_weights, best_error

final_weights, test_error = pocket_algorithm(training_features, training_labels, max_updates=1000)
print(f'Test Error: {100 * test_error:.2f}%')

# Plot the data points and the hypothesis line
plt.figure(figsize=(8, 6))
plt.scatter(test_features[:, 1], test_features[:, 2], c=test_labels, s=20)
x = np.linspace(-1, 1, 100)
if abs(final_weights[3]) > 1e-6:
    y = (-final_weights[1] * x - final_weights[0] - final_weights[2] * x ** 3) / final_weights[3]
else:
    y = np.zeros_like(x)
plt.plot(x, y, '-g', label='Hypothesis Line')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Test Data with Pocket Algorithm Hypothesis')
plt.show()