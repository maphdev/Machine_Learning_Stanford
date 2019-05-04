import numpy as np

# Compute the cost for linear regression of using theta as the parameter
# for linear regression to fit the data points in X and y
def computeMeanSquaredErrorCost(X, y, theta):
    m = y.size
    J = (1/(2*m)) * np.sum(np.power((X.dot(theta) - y), 2))
    return J

# Performs gradient descent to learn theta by taking num_iters gradient
# steps with learning rate alpha
def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        theta = theta - alpha * (1/m) * X.transpose().dot((X.dot(theta) - y))
        J_history[i] = computeMeanSquaredErrorCost(X, y, theta)
    return [theta, J_history]

# Computes the closed-form solution to linear regression using the normal
# equations
def normalEquation(X, y):
    theta = np.linalg.pinv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
    return theta

# Returns a normalized version of X where the mean value of each feature
# is 0 and the standard deviation is 1. This is often a good preprocessing
# step to do when working with learning algorithms
def normalizeFeature(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1)
    X_norm = np.divide(X-mu, sigma)
    return [X_norm, mu, sigma]
