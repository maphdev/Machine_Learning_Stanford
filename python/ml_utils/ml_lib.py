import numpy as np

# Compute the cost for linear regression of using theta as the parameter
# for linear regression to fit the data points in X and y
def meanSquaredErrorCost(X, y, theta):
    m = y.size
    J = (1/(2*m)) * np.sum(np.power((X.dot(theta) - y), 2))
    return J

# Performs gradient descent to learn theta by taking num_iters gradient
# steps with learning rate alpha
def linearRegressionGradientDescent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        theta = theta - alpha * (1/m) * X.transpose().dot((X.dot(theta) - y))
        J_history[i] = meanSquaredErrorCost(X, y, theta)
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

# Computes the sigmoid of z.
def sigmoid(z):
    return 1 / (1+np.exp(-z))

# Computes the cost of using theta as the parameter for logistic regression
# and the gradient of the cost w.r.t. to the parameters.
# Possibility to get the regularized cost.
def logLossCost(X, y, theta, regularized=False, lbd=1):
    m = y.size
    z = X.dot(theta)
    h_x = sigmoid(z)
    if regularized:
        J = (1/m)*sum(-y*np.log(h_x)-(1-y)*np.log(1-h_x)) + (lbd/(2*m))*sum(theta[1:]**2)
    else:
        J = (1/m)*sum(-y*np.log(h_x)-(1-y)*np.log(1-h_x))
    return J

# Performs gradient descent to learn theta by taking num_iters gradient steps
# with learning rate alpha
# TO DO : regularized version
def logisticRegressionGradientDescent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        z = X.dot(theta)
        h_x = sigmoid(z)
        theta = theta - alpha * (1/m) * X.transpose().dot((h_x - y))
        #grad = (1/m) * np.sum(np.multiply(h_x - y, X), axis=0)
        J_history[i] = logLossCost(X, y, theta)
    return [theta, J_history]

# Predict whether the label is 0 or 1 using learned logistic regression
# parameters theta
def predictLogisticRegression(X, theta):
    h_x = sigmoid(X.dot(theta))
    return (h_x >= 0.5)

# Maps the two input features to quadratic features
def mapPolynomialFeature(x1,x2,degree):
    out = np.ones(len(x1)).reshape(len(x1),1)
    for i in range(1,degree+1):
        for j in range(i+1):
            terms= (x1**(i-j) * x2**j).reshape(len(x1),1)
            out= np.hstack((out,terms))
    return out
