'''
// Main File:        regression.py
// Semester:         CS 540 Fall 2020
// Authors:          Tae Yong Namkoong
// CS Login:         namkoong
// References:       TA's & Peer Mentor's Office Hours
                     https://www.geeksforgeeks.org/linear-regression-python-implementation/
                     https://www.geeksforgeeks.org/solving-linear-regression-in-python/
                     https://towardsdatascience.com/implement-gradient-descent-in-python-9b93ed7108d1
                     https://stackoverflow.com/questions/17784587/gradient-descent-using-python-and-numpy
                     https://towardsdatascience.com/simple-and-multiple-linear-regression-with-python-c9ab422ec29c
'''
import csv
import numpy as np
import math
import random
from numpy.linalg import inv


def get_dataset(filename):
    '''
    takes a filename and returns the data as described below in an n-by-(m+1) array
    '''
    data = []
    with open(filename, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(csvfile)
        for row in reader:
            row = row[1:]
            data.append(row)
    output = np.array(data)
    return output.astype(np.float)

def print_stats(dataset, col):
    '''
    takes the dataset as produced by the previous function and prints several statistics about a column of the dataset;
    does not return anything
    '''
    data = (dataset[:,col])
    print(data.size, end='\n')
    calc_mean = np.mean(data)
    calc_std = np.std(data)
    print("{:.2f}".format(calc_mean), end='\n')
    print("{:.2f}".format(calc_std), end='\n')

def regression(dataset, cols, betas) :
    '''
    calculates and returns the mean squared error on the dataset given fixed betas
    '''
    matrix = []
    matrix.append(dataset[:,0])
    mse = 0
    for col in cols:
        data = (dataset[:,col])
        matrix.append(data)
    matrix = np.array(matrix)
    size = len(dataset)
    for i in range(size):
        fx = betas[0]
        for j in range(len(cols)):
            fx = fx + betas[j+1] * matrix[j+1][i]
        mse = mse + (fx - matrix[0][i])**2
    mse = (1/size) * mse
    return mse

def gradient_descent(dataset, cols, betas):
    '''
    performs a single step of gradient descent on the MSE and returns the derivative values as an 1D array
    '''
    x = []
    y = dataset[:, 0]
    gradient = np.zeros(len(betas))
    for col in cols:
        data = (dataset[:, col])
        x.append(data)
    col_length = len(cols)
    for i in range(len(y)):
        partial_derv = betas[0]
        for j in range(col_length):
            partial_derv = partial_derv + betas[j + 1] * x[j][i]
        partial_derv = partial_derv - y[i]
        gradient[0] = partial_derv + gradient[0]
        for k in range(col_length):
            gradient[k + 1] = gradient[k + 1] + (partial_derv * x[k][i])
    gradient = (2 / len(y)) * gradient
    return gradient

def iterate_gradient(dataset, cols, betas, T, eta):
    '''
    performs T iterations of gradient descent starting at the given betas and prints the results; does not return anything
    '''
    betas_list = betas
    for i in range(1, T + 1):
        print(i, end=' ')
        gradient_beta_0 = gradient_descent(dataset, cols, betas_list)

        for j in range(len(betas_list)):
            betas_list[j] = betas_list[j] - (eta * gradient_beta_0[j])
            # print(betas)
            mse = regression(dataset, cols, betas_list)
        print("{:.2f}".format(mse), end=' ')
        for k in range(len(betas_list)):
            print("{:.2f}".format(betas_list[k]), end=' ')
        print(end='\n')

def compute_betas(dataset, cols):
    '''
    using the closed-form solution, calculates and returns the values of betas and the corresponding MSE as a tuple
    '''
    x = []
    y = dataset[:, 0]
    for i in range(len(y)):
        x.append([1])
        for col in cols:
            x[i].append(dataset[i][col])
    x = np.matrix(x)
    part1 = inv(x.transpose().dot(x))
    part2 = x.transpose().dot(y)
    betas = np.squeeze(np.asarray(part1).dot(np.squeeze(np.asarray(part2))))
    mse = regression(dataset,cols,betas)
    return (mse, *betas)

def predict(dataset, cols, features):
    '''
    using the closed-form solution betas, return the predicted body fat percentage of the give features.
    '''
    (mse, *betas) = compute_betas(dataset,cols)
    final = betas[0]
    for i in range(len(betas)-1):
        final = final + (betas[i+1] * features[i])
    return(final)

def sgd(dataset, cols, betas, T, eta):
    """
    performs stochastic gradient descent, prints results as in function 5
    """
    random_generator = random_index_generator(0, len(dataset))
    betas_list = betas

    for i in range(1, T+1):
        print(i, end=' ')
        rand_no = next(random_generator)
        grads = get_rand_grad(dataset,betas_list, cols, rand_no)
        beta_length = len(betas_list)
        for j in range(beta_length):
            betas_list[j] = betas_list[j] - eta * grads[j]
            mse = regression(dataset, cols, betas_list)
        print("{:.2f}".format(mse), end=' ')
        for k in range(len(betas_list)):
            print("{:.2f}".format(betas_list[k]), end=' ')
        print(end='\n')

def get_rand_grad(dataset, betas, cols, rand_no):
    x = []
    y = dataset[:,0]
    gradients = np.zeros(len(betas))
    rand = rand_no
    for col in cols:
        data = (dataset[:,col])
        x.append(data)
    partial_derv = betas[0]
    col_length = len(cols)
    for j in range(col_length):
        partial_derv = partial_derv + (betas[j+1] * x[j][rand])
    partial_derv = partial_derv - y[rand]
    gradients[0] = gradients[0] + partial_derv
    for k in range(col_length):
        gradients[k+1] = gradients[k+1] + (partial_derv * x[k][rand])
    return 2 * gradients

def random_index_generator(min_val, max_val, seed=42):
    """
    DO NOT MODIFY THIS FUNCTION.    DO NOT CHANGE THE SEED.
    This generator picks a random value between min_val and max_val,    seeded by 42.
    """
    random.seed(seed)
    while True:
        yield random.randrange(min_val, max_val)
