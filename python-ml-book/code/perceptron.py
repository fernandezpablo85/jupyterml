from __future__ import division
import numpy as np

def net_input(xi, weights):
    return np.dot(xi, weights[1:]) + weights[0]

def predict(xi, weights):
    return np.where(net_input(xi, weights) >= 0, 1, -1)

def fit(X, y, learning_rate=0.01, iterations=10):
    number_of_features = X.shape[1]
    weights = np.zeros(number_of_features + 1)  # +1 for bias weight
    total_errors = []
    for _ in range(iterations):
        iteration_errors = 0
        for xi, yi in zip(X, y):
            predicted = predict(xi, weights)
            weights[1:] += learning_rate * (yi - predicted) * xi
            weights[0] += learning_rate * (yi - predicted)
            if predicted != yi:
                iteration_errors += 1
        total_errors.append(iteration_errors)
    return weights, total_errors
