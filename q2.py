from scipy import io
import pandas as pd
import matplotlib.pyplot as plt
from sympy import *
import numpy as np
from math import e

mat = io.loadmat("data3-2.mat")
circles = mat['circles']
stars = mat['stars']
s_pd = pd.DataFrame(stars, columns=['x', 'y'])
c_pd = pd.DataFrame(circles, columns=['x', 'y'])
s_pd['Label'] = 1
c_pd['Label'] = -1
data = pd.concat([s_pd, c_pd])
data = data.reset_index()
# plt.figure(figsize=(15, 10))
# plt.plot(s_pd['x'], s_pd['y'], "*")
# plt.plot(c_pd['x'], c_pd['y'], "o")


def formulate_gaussian_kernel(h, x, x_):
  exponent = -((x[0] - x_[0])**2 + (x[1] - x_[1])**2)/(h)
  numerator = e**exponent
  return numerator

def formulate_simple_kernel(x, x_):
    ans = 1 + x[0]*x_[0] + x[1]*x_[1]
    return ans**2

def compute_coefficients(data, regularizer, h=1, kernel_type="Gaussian"):
  k = np.zeros((len(data), len(data)))
  if kernel_type == "Gaussian":
    for i in range(0, len(data)):
      for j in range(0, len(data)):
        k[i][j] = formulate_gaussian_kernel(h, (data.iloc[i]['x'], data.iloc[i]['y']), \
                                          (data.iloc[j]['x'], data.iloc[j]['y']))
  elif kernel_type == "Simple":
    for i in range(0, len(data)):
      for j in range(0, len(data)):
        k[i][j] = formulate_simple_kernel((data.iloc[i]['x'], data.iloc[i]['y']), \
                                        (data.iloc[j]['x'], data.iloc[j]['y']))

  regularizer_identity = regularizer * np.identity(len(data))
  inverse = np.linalg.inv(regularizer_identity + k)
  label = np.array(data['Label']).reshape((-1, 1))
  return np.matmul(inverse, label)

def compute_gaussian_kernel(h, x_):
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    exponent = -((x - x_[0]) ** 2 + (y - x_[1]) ** 2) / (h)
    numerator = e**exponent
    return sympify(numerator)


def compute_simple_kernel(x_):
  x = Symbol('x', real=True)
  y = Symbol('y', real=True)
  ans = 1 + x * x_[0] + y * x_[1]
  return ans ** 2

def separating_boundary_function(coefficients, h=1, kernel_type="Gaussian"):
    g_x = 0
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    if kernel_type == "Gaussian":
      for i in range(0, len(data)):
        g_x = g_x + coefficients[i] * \
              compute_gaussian_kernel(h, (data.iloc[i]['x'], data.iloc[i]['y']))
    elif kernel_type == "Simple":
      for i in range(0, len(data)):
        g_x = g_x + coefficients[i] * \
              compute_simple_kernel((data.iloc[i]['x'], data.iloc[i]['y']))
    return sympify(g_x)[0]

def training_accuracy(data, g_x, h, regularizer):
  x = Symbol('x', real = True)
  y = Symbol('y', real = True)
  misclassified = 0
  for i in range(0, len(data)):
    if (g_x.subs({x:data.iloc[i]['x'], y:data.iloc[i]['y']})) > 0:
      if data.iloc[i]['Label'] == -1:
        misclassified = misclassified + 1
    else:
      if data.iloc[i]['Label'] == 1:
        misclassified = misclassified + 1
  print("Misclassified points number: ", misclassified, "/42")
  print("Accuracy %:", misclassified/len(data)*100)
  print("For h:", h, " for regularizer:", regularizer)

def plot_boundary(data, regularizer=0.1, h=1, \
                  support_x=(-1.25, 1.25), support_y=(-0.5, 1.5), \
                  kernel_type="Gaussian", precision_x = 40, \
                 precision_y = 40):
    a = compute_coefficients(data, regularizer, h, kernel_type)
    g_x = separating_boundary_function(a, h, kernel_type)
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)

    y_val = []
    x_axis = []
    for x_ in range(int(support_x[0]*precision_x), int(support_x[1]*precision_x), 1):
      for y_ in range(int(support_y[0]*precision_y), int(support_y[1]*precision_y), 1):
        if (g_x.subs({x:x_/precision_x, y:y_/precision_y}) > 0):
          y_val.append(y_/precision_y)
          x_axis.append(x_/precision_x)
          break

#     y_val2 = []
#     x_axis2 = []
#     for x_ in range(int(support_x[0]*precision_x), int(support_x[1]*precision_x), 1):
#         flag = 0
#         for y_ in range(int(support_y[1]*precision_y), int(support_y[0]*precision_y), -1):
#             if (g_x.subs({x:x_/precision_x, y:y_/precision_y}) < 0) and (flag == 0):
#                 y_val2.append(y_/precision_y)
#                 x_axis2.append(x_/precision_x)
#                 break
    training_accuracy(data, g_x, h, regularizer)
    plt.figure(figsize=(15, 10))
    plt.plot(s_pd['x'], s_pd['y'], "*")
    plt.plot(c_pd['x'], c_pd['y'], "o")
    plt.plot(x_axis, y_val, "-o", label="boundary - h=0.0001, reg=0.01, kernel=Simple")
#     plt.plot(x_axis2, y_val2, "-o", label="boundary2")
    plt.legend()
    plt.show()

    return g_x

g_x = plot_boundary(data, support_x=(-1.5, 1.5), \
                    support_y=(-0.5, 1.5), h=0.0001, regularizer=0.01, \
                    precision_x=40, precision_y=50, kernel_type="Simple")
