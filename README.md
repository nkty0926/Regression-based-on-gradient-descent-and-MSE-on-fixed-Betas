Hybrid Machine Learning Optimization Model for Predicting Body Fat Percentages By Using Stepwise Iterative/Stochastic Gradient Descents on Mean Squared Error on a Given Dataset of Features by Fixing Betas in a Closed-Form Solution.

In the context of Machine Learning Optimization, some features may be sparse. If the mean gradient for sparse features is minimal, optimizations may occur at a much lower training rate. It is possible to optimize the training rate by iteratively taking steps in the opposite direction of the graident. By performing T itereations of gradient descent on MSE (mean squared errors), this program computes gradient descent on the MSE by fixing betas to compute the corresponding derivative values as an 1D array.

Gradient descent is a first-order iterative optimization algorithm to minimize a loss function, thereby  a local minimum of a differentiable function. 

##Import Libraries 

```
from regression import get_dataset, regression, iterate_gradient, gradient_descent, compute_betas, predict, sgd, print_stats
import csv
import numpy as np
import math
import random
import unittest
from numpy.linalg import inv
```

![](Images/csv%20file.png)
![](Images/test%20results1.png)
![](Images/test%20results2.png)
