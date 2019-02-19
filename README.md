# Kernel Application in Density Estimation and Classification

There are 2 files.

One uses gaussian and laplacian kernels to estimate the density function of random variables sampled from uniform distribution.

Second uses gaussian kernels to classify points that were not linearly separable.

### Prerequisites

```
pip install sympy
```

### q1 Parameters:

Parameters for q1:
* outer_support = tuple of int. defines the region of x-axis where we want the function to be plotted.
* trials = int. number of random variables to be generated from the given distribution.
* function_type = "Uniform". the type of distribution to predict. As of now, only "Uniform" is supported.
* inner_support = tuple of int. defines the region of x-axis where we want our estimate to be plotted.
* kernel_type = "Gaussian"/"Laplacian". the type of kernel we want to use in our estimator.
* h = float. the variance of our estimator
* precision = int. the number of points on x-axis and y-axis for which we calculate our function and estimate.


### q2 Parameters:

Parameters for q2:
* outer_support = tuple of int. defines the region of x-axis where we want the function to be plotted.
* trials = int. number of random variables to be generated from the given distribution.
* function_type = "Uniform". the type of distribution to predict. As of now, only "Uniform" is supported.
* inner_support = tuple of int. defines the region of x-axis where we want our estimate to be plotted.
* kernel_type = "Gaussian"/"Laplacian". the type of kernel we want to use in our estimator.
* h = float. the variance of our estimator
* precision_x, precision_y = int. the number of points on x-axis and y-axis for which we calculate our function and estimate.
* regularizer = float. learning rate
* kernel_type = "Simple"/"Gaussian". Simple kernel is a quadratic kernel. Choice of kernel to use for our classifier.

### Running the Application

```
python q1.py
python q2.py
```

## Sample Graphs q1

Using Gaussian Kernel to estimate the actual distribution:\
![Alt text](q1_gaussian_h0_01.png?raw=true "q1_gaussian_h0_01")
![Alt text](q1_gaussian_h0_001.png?raw=true "q1_gaussian_h0_001")

The rest of the graphs are uploaded with prefix "q1".

## Sample Graphs q2

Classification of blue stars and orange circles:\
![Alt text](q2_gaussian_h0_1_reg0_1.png?raw=true "q2_gaussian_h0_1_reg0_1")
![Alt text](q2_gaussian_h0_01_reg0_1.png?raw=true "q2_gaussian_h0_01_reg0_1")

The rest of the graphs are uploaded with prefix "q2".
