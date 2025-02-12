# -*- coding: utf-8 -*-
"""ML_Homework-3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oSnwrEDTSZ-sumqSASGXkeCdLRRmwEPL
"""

# Task-2 [HP1]

# Part 1

import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derv(x):
    return 1 - np.tanh(x)**2


def squared_error(y_true, y_pred):
    return 0.5 * np.sum((y_true - y_pred)**2)

def forward_prop(X, W1, b1, W2, b2):
    # Input layer (Layer 0)
    a0 = X

    # Hidden layer (Layer 1)
    z1 = np.dot(a0, W1) + b1
    a1 = tanh(z1)

    # Output layer (Layer 2)
    z2 = np.dot(a1, W2) + b2
    a2 = tanh(z2)

    return a0, z1, a1, z2, a2


def backward_prop(X, y, a0, z1, a1, z2, a2, W1, W2):
    m = X.shape[0]

    # Output layer (Layer 2)
    delta2 = (a2 - y) * tanh_derv(z2)

    # Hidden layer (Layer 1)
    delta1 = np.dot(delta2, W2.T) * tanh_derv(z1)

    # Gradients
    dW2 = np.dot(a1.T, delta2) / m
    db2 = np.sum(delta2, axis=0) / m
    dW1 = np.dot(a0.T, delta1) / m
    db1 = np.sum(delta1, axis=0) / m

    return dW1, db1, dW2, db2

# Part 2

import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derv(x):
    return 1 - np.tanh(x)**2

def forward_prop(X, W1, b1, W2, b2):
    # Input layer (Layer 0)
    a0 = X

    # Hidden layer (Layer 1)
    z1 = np.dot(a0, W1) + b1
    a1 = tanh(z1)

    # Output layer (Layer 2)
    z2 = np.dot(a1, W2) + b2
    a2 = tanh(z2)

    return a0, z1, a1, z2, a2

def squared_error(y_true, y_pred):
    return 0.5 * np.sum((y_true - y_pred)**2)

def backward_prop(X, y, a0, z1, a1, z2, a2, W1, W2):
    m = X.shape[0]

    # Output layer (Layer 2)
    delta2 = (a2 - y) * tanh_derv(z2)

    # Hidden layer (Layer 1)
    delta1 = np.dot(delta2, W2.T) * tanh_derv(z1)

    # Gradients
    dW2 = np.dot(a1.T, delta2) / m
    db2 = np.sum(delta2, axis=0, keepdims=True) / m
    dW1 = np.dot(a0.T, delta1) / m
    db1 = np.sum(delta1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

def train_neural_network(X, y, num_epochs, learning_rate, m, batch_size=1):
    # Initialize weights and biases
    np.random.seed(0)
    W1 = np.random.randn(2, m)
    b1 = np.zeros((1, m))
    W2 = np.random.randn(m, 1)
    b2 = np.zeros((1, 1))

    for epoch in range(num_epochs):
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            # Forward Propagation
            a0, z1, a1, z2, a2 = forward_prop(X_batch, W1, b1, W2, b2)

            # Compute the loss
            loss = squared_error(y_batch, a2)

            # Backward Propagation
            dW1, db1, dW2, db2 = backward_prop(X_batch, y_batch, a0, z1, a1, z2, a2, W1, W2)

            # Update weights and biases
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        if (epoch + 1) % 1000 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss}')

    return W1, b1, W2, b2

# Part 3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt

# Load training and testing data
train_data = pd.read_csv('ZipDigits.train', sep=' ')
test_data = pd.read_csv('ZipDigits.test', sep=' ')

# Convert the first column to integer
train_data[train_data.columns[0]] = train_data[train_data.columns[0]].astype(int)
test_data[test_data.columns[0]] = test_data[test_data.columns[0]].astype(int)

# Remove the last column (if it exists) as it seems to be an extra index
if len(train_data.columns) > 1:
    train_data = train_data.iloc[:, :-1]

if len(test_data.columns) > 1:
    test_data = test_data.iloc[:, :-1]

# Convert the first column to integer type
train_data[train_data.columns[0]] = train_data[train_data.columns[0]].astype(int)
test_data[test_data.columns[0]] = test_data[test_data.columns[0]].astype(int)

# Filter for digits 1 and 5, and encode as 1 and -1 respectively
train_data = train_data[(train_data[train_data.columns[0]] == 1) | (train_data[train_data.columns[0]] == 5)]
train_data[train_data.columns[0]] = np.where(train_data[train_data.columns[0]] == 1, 1, -1)

test_data = test_data[(test_data[test_data.columns[0]] == 1) | (test_data[test_data.columns[0]] == 5)]
test_data[test_data.columns[0]] = np.where(test_data[test_data.columns[0]] == 1, 1, -1)

# Extract features and labels
X_train = train_data.iloc[:, 1:]
y_train = train_data[train_data.columns[0]]

X_test = test_data.iloc[:, 1:]
y_test = test_data[test_data.columns[0]]

X_train = X_train.values.reshape(-1, 16, 16)
X_test = X_test.values.reshape(-1, 16, 16)
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

# Feature extraction functions
def calculate_intensity(gray_image):
    intensity = np.mean(gray_image)
    return intensity

def calculate_symmetry(gray_image):
    flipped_image = cv.flip(gray_image, 1)
    difference = cv.absdiff(gray_image, flipped_image)
    symmetry = np.mean(difference)
    return symmetry

# Calculate intensities and symmetries for training data
train_intensities = [calculate_intensity(image) for image in X_train]
train_symmetries = [calculate_symmetry(image) for image in X_train]

# Calculate intensities and symmetries for testing data
test_intensities = [calculate_intensity(image) for image in X_test]
test_symmetries = [calculate_symmetry(image) for image in X_test]

# Add calculated features to the dataframes
train_data['intensity'] = train_intensities
train_data['symmetry'] = train_symmetries
test_data['intensity'] = test_intensities
test_data['symmetry'] = test_symmetries

# Create new dataframes for training and testing
train_df = pd.DataFrame({
    'x0': 1,
    'x1': train_data['intensity'],
    'x2': train_data['symmetry'],
    'y': train_data[train_data.columns[0]]  # Assuming the target column is the first column
})

test_df = pd.DataFrame({
    'x0': 1,
    'x1': test_data['intensity'],
    'x2': test_data['symmetry'],
    'y': test_data[test_data.columns[0]]  # Assuming the target column is the first column
})

# Activation function and its derivative
def tanh_activation(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2

# Initialize weights
weights1 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
weights2 = np.array([[1], [2], [3]])

# Forward propagation function
def forward_propagation(weights1, weights2, inputs):
    weights1_t = weights1.T
    weights2_t = weights2.T

    s1 = np.dot(weights1_t, inputs)
    x1 = np.insert(tanh_activation(s1), 0, 1, axis=0)

    s2 = np.dot(weights2_t, x1)
    x2 = np.insert(tanh_activation(s2), 0, 1, axis=0)

    output = tanh_activation(s2)

    return x1, x2, s1, s2, weights1, weights2, output

def back_propagation(x1, s2, weights1, weights2, target, inputs):
    learning_rate = 0.01

    delta2 = 2 * (tanh_activation(s2) - target) * tanh_derivative(s2)
    delta1 = delta2 * weights2[1:, 0] * tanh_derivative(s1)

    gradient2 = np.outer(delta2, x1)
    gradient1 = np.outer(delta1, inputs)

    final_weights2 = weights2 - learning_rate * gradient2.T
    final_weights1 = weights1 - learning_rate * gradient1.T

    return final_weights1, final_weights2

# Set the number of epochs
epochs = 100

# Convert DataFrame to numpy array
train_inputs = train_df[['x0', 'x1', 'x2']].values.T
train_targets = train_df['y'].values.reshape(1, -1)

# Training the neural network
for epoch in range(epochs):
    for i in range(train_inputs.shape[1]):
        x1, x2, s1, s2, weights1, weights2, output = forward_propagation(weights1, weights2, train_inputs[:, i])
        weights1, weights2 = back_propagation(x1, s2, weights1, weights2, train_targets[0, i], train_inputs[:, i])

# Part-4

def train_neural_network(train_df, w1, w2, epochs=1000):
    loss_list = []

    for epoch in range(epochs):
        for i in range(len(train_df)):
            random_index = np.random.randint(0, len(train_df))
            inputs = np.array([1, train_df.iloc[random_index, 1], train_df.iloc[random_index, 2]])
            target = train_df.iloc[random_index, 3]

            x1, x2, s1, s2, w1, w2, output = forward_propagation(w1, w2, inputs)

            final_w1, final_w2 = back_propagation(x1, s2, w1, w2, target, inputs)

            w1 = final_w1
            w2 = final_w2

            neural_network_answer = tanh_activation(final_w2[1][0])

            if target == 1:
                loss = np.mean(np.square(target - neural_network_answer))
            else:
                loss = np.mean(np.square(neural_network_answer))

            if i == len(train_df) - 1 and epoch % 100 == 0:
                print('Epoch : ', epoch, ', Loss : ', loss)

        loss_list.append(loss)

    return w1, w2, loss_list

# Set the number of epochs
epochs = 1000

# Initialize weights
w1 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
w2 = np.array([[1], [2], [3]])

# Train the neural network
final_w1, final_w2, loss_list = train_neural_network(train_df, w1, w2, epochs)

# Plotting the loss
plt.plot(range(0, epochs), loss_list)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.show()

def test_neural_network(test_df, w1, w2):
    correct_count = 0

    for i in range(len(test_df)):
        inputs = np.array([1, test_df.iloc[i, 1], test_df.iloc[i, 2]])
        target = test_df.iloc[i, 3]

        _, _, _, _, _, _, output = forward_propagation(w1, w2, inputs)

        if np.sign(output) == np.sign(target):
            correct_count += 1

    accuracy = correct_count / len(test_df) * 100

    return accuracy

# Test the neural network on the test data
test_accuracy = test_neural_network(test_df, final_w1, final_w2)
print('Test Accuracy:', test_accuracy)

# Part-5

def predict(inputs, weights1, weights2):
    _, _, _, _, _, _, output = forward_propagation(weights1, weights2, inputs)
    return np.sign(output)


# Calculate Ein (in-sample error)
train_inputs = train_df[['x0', 'x1', 'x2']].values.T
train_targets = train_df['y'].values.reshape(1, -1)

train_predictions = np.apply_along_axis(predict, axis=0, arr=train_inputs, weights1=final_w1, weights2=final_w2)
Ein = np.mean(train_predictions != train_targets)

# Calculate test error
test_inputs = test_df[['x0', 'x1', 'x2']].values.T
test_targets = test_df['y'].values.reshape(1, -1)

test_predictions = np.apply_along_axis(predict, axis=0, arr=test_inputs, weights1=final_w1, weights2=final_w2)
test_error = np.mean(test_predictions != test_targets)

# Report Ein and test error
print('Ein (in-sample error):', Ein)
print('Test error:', test_error)

# HP2

# Mini-batch gradient descent training loop
batch_size = 32
epochs = 1000
custom_loss_list = []
ein_list = []

for epoch in range(epochs):
    for _ in range(0, len(train_df), batch_size):
        batch_indices = np.random.choice(len(train_df), size=batch_size, replace=False)
        batch_inputs = train_df.iloc[batch_indices, 1:3].values
        batch_targets = train_df.iloc[batch_indices, 3].values

        for i in range(batch_size):
            inputs = np.insert(batch_inputs[i], 0, 1)
            target = batch_targets[i]

            x1, x2, s1, s2, w1, w2, output = forward_propagation(w1, w2, inputs)

            final_w1, final_w2 = back_propagation(x1, s2, w1, w2, target, inputs)

            w1 = final_w1
            w2 = final_w2

            neural_network_answer = tanh_activation(final_w2[1][0])

            if target == 1:
                loss = np.mean(np.square(target - neural_network_answer))
            else:
                loss = np.mean(np.square(neural_network_answer))

            custom_loss_list.append(loss)

    if epoch % 100 == 0:
        print('Epoch : ', epoch, ', Loss : ', loss)

# Calculate Ein (in-sample error) after mini-batch gradient descent
train_inputs = train_df[['x0', 'x1', 'x2']].values.T
train_targets = train_df['y'].values.reshape(1, -1)

train_predictions = np.apply_along_axis(predict, axis=0, arr=train_inputs, weights1=final_w1, weights2=final_w2)
Ein = np.mean(train_predictions != train_targets)
ein_list.append(Ein)

# Calculate test error after mini-batch gradient descent
test_inputs = test_df[['x0', 'x1', 'x2']].values.T
test_targets = test_df['y'].values.reshape(1, -1)

test_predictions = np.apply_along_axis(predict, axis=0, arr=test_inputs, weights1=final_w1, weights2=final_w2)
test_error = np.mean(test_predictions != test_targets)

# Report Ein and test error after mini-batch gradient descent
print('Ein (in-sample error) after mini-batch gradient descent:', Ein)
print('Test error after mini-batch gradient descent:', test_error)


# Plot Ein over epochs after mini-batch gradient descent
plt.plot(range(0, epochs, 100), ein_list[::100])  # Plot every 100 epochs for better visualization
plt.xlabel('Epochs')
plt.ylabel('Ein')
plt.title('In-sample Error (Ein) Over Epochs after Mini-batch Gradient Descent')
plt.show()