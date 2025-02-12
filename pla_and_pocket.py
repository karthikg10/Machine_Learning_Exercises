# -*- coding: utf-8 -*-
"""PLA and Pocket.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1b45WI6xqhA7_XABD-2Poos_Xi-PlyCoN
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('synthetic_dataset.csv')
x = dataset[['x1', 'x2']].values
y = dataset['y'].values

def perceptron(x,y,max_iters =1000):
  w = np.zeros(x.shape[1])
  bias = 0


  for _ in range(max_iters):
    misclassified = 0
    for i in range(len(x)):
      if y[i] * (np.dot(x[i], w) + bias) <=0:
        w += y[i] * x[i]
        bias += y[i]
        misclassified +=1
    if not misclassified:
      break

  return w, bias

w, bias = perceptron(x,y)

plt.scatter(x[:, 0], x[:, 1], c=y)
x_boundary = np.array([x[:, 0].min(), x[:, 0].max()])
y_boundary_pla = (-bias - w[0] * x_boundary) / w[1]
plt.plot(x_boundary, y_boundary_pla, label='PLA Decision Boundary', color='r')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Dataset with PLA Decision Boundary')
plt.show()

def pocket(x,y, max_iters):
  w = np.zeros(x.shape[1])
  bias = 0
  pocket_w = w.copy()

  for _ in range(max_iters):
    misclassified = 0
    for i in range(len(x)):
      if y[i] * (np.dot(x[i], w) + bias)<= 0:
        w += y[i] * x[i]
        bias += y[i]
        misclassified +=1
        if misclassified >0:
          pocket_w = w.copy()

    if misclassified == 0:
      break

  return pocket_w, bias

pocket_w, bias = perceptron(x,y)

plt.scatter(x[:, 0], x[:, 1], c=y)
x_boundary = np.array([x[:, 0].min(), x[:, 0].max()])
y_boundary_pla = (-bias - w[0] * x_boundary) / w[1]
plt.plot(x_boundary, y_boundary_pla, label='Pocket Decision Boundary', color='r')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Dataset with Pocket Decision Boundary')
plt.show()