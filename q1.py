import numpy as np
import random
from sympy import *
import math
import matplotlib.pyplot as plt

def uniform_dist_instances(number, a, b):
  vec = [random.uniform(a, b) for _ in range(0, number)]
  return vec

def compute_gaussian_kernel(h, x_obs):
  x = Symbol('x', real=True)
  exponent = -((x - x_obs)**2)/(2*h)
  numerator = exp(exponent)
  denominator = math.sqrt(2*math.pi*h)
  return numerator/denominator

def compute_laplacian_kernel(h, x_obs):
  x = Symbol('x', real=True)
  exponent = -(Abs(x - x_obs))/(h)
  numerator = exp(exponent)
  denominator = math.sqrt(2*h)
  return numerator/denominator

def approximate_function(vec, h, kernel_type = "Gaussian"):
  sum = 0
  if kernel_type == "Gaussian":
    for i in range(0, len(vec)):
      sum = sum + compute_gaussian_kernel(h, vec[i])
  elif kernel_type == "Laplacian":
    for i in range(0, len(vec)):
      sum = sum + compute_laplacian_kernel(h, vec[i])
  return sum/len(vec)

def plot_approximate_function(outer_support = (-2, 2), \
                              trials=100, function_type="Uniform", inner_support=(0, 1), \
                              kernel_type = "Gaussian", h = 0.01, precision = 100):
  if function_type == "Uniform":
    vec = uniform_dist_instances(trials, inner_support[0], inner_support[1])

  approx_function = approximate_function(vec, h, kernel_type)
  x = Symbol('x', real=True)
  val = [approx_function.subs({x:(x_i/precision)}) \
         for x_i in range(outer_support[0]*precision, \
         outer_support[1]*precision, 1)]

  plt.figure(figsize=(15, 10))
  x_axis = [x/precision for x in range(outer_support[0]*precision, \
                                         outer_support[1]*precision, 1)]
  plt.plot(x_axis, val, "-o", label = "approximate")

  actual_precision = 1000
  actual_y = [1 for _ in range(0, actual_precision)]
  actual_x = [x/actual_precision for x in range(0*actual_precision, \
                                                1*actual_precision, 1)]
  plt.plot(actual_x, actual_y, "-", label = "actual")

  plt.ylabel("density")
  plt.legend()
  plt.show()

plot_approximate_function(trials=1000, kernel_type="Laplacian", h=0.0001, precision=100)
