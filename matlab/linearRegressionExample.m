% Path
addpath('ml_utils/')

% ==============================================================================

fprintf('\n------------------------------------------------------------\n');
fprintf('LINEAR REGRESSION WITH ONE FEATURE\n');
fprintf('------------------------------------------------------------\n\n');

% Load the dataset
data = load('../data/data1.txt');

% ==============================================================================

% Select X and Y
X = data(:, 1);
y = data(:, 2);

% Number of observations
m = length(y);

% Treat the intercept term as another feature
X = [ones(m, 1), X];

% Theta initialization
theta = zeros(2, 1);

% Parameters
alpha = 0.01;
num_iters = 1500;

% ==============================================================================

fprintf('Cost at initialization : %0.2f\n\n', meanSquaredErrorCost(X, y, theta));

% ==============================================================================

% Thetas with gradient descent
[theta, history] = linearRegressionGradientDescent(X, y, theta, alpha, num_iters);

% ==============================================================================

fprintf('Theta computed from gradient descent:\n%f, \n%f\n\n',theta(1),theta(2));

% ==============================================================================

fprintf('Cost after gradient descent : %0.2f\n\n', meanSquaredErrorCost(X, y, theta));

% ==============================================================================

predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n\n', predict1*10000);

predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n\n', predict2*10000);

% ==============================================================================

fprintf('------------------------------------------------------------\n');
fprintf('LINEAR REGRESSION WITH MULTIPLE FEATURE\n');
fprintf('------------------------------------------------------------\n\n');

data = load('../data/data2.txt');

% ==============================================================================

% Select train and test dataset
X = data(:, 1:2);
y = data(:, 3);

% Number of observations
m = length(y);

% Normalize (save time for gradient descent)
[X, mu, sigma] = normalizeFeature(X);

% Treat the intercept term as another feature
X = [ones(m, 1) X];

% Theta initialization
theta = zeros(3, 1);

% Parameters
alpha = 0.1;
num_iters = 400;

% ==============================================================================

fprintf('Cost at initialization : %0.2f\n\n', meanSquaredErrorCost(X, y, theta));

% ==============================================================================

% Thetas with gradient descent
[theta, hist] = linearRegressionGradientDescent(X, y, theta, alpha, num_iters);

% ==============================================================================

fprintf('Theta computed from gradient descent:\n%f, \n%f, \n%f\n\n',theta(1),theta(2), theta(3));

% ==============================================================================

fprintf('Cost after gradient descent : %0.2f\n\n', meanSquaredErrorCost(X, y, theta));

% ==============================================================================

test = [1650 3];
normalizedTest = (test-mu)./sigma;
test = [1, normalizedTest];
predict3 = test*theta;

fprintf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n$%f\n\n', predict3);

% ==============================================================================

fprintf('------------------------------------------------------------\n');
fprintf('LINEAR REGRESSION WITH NORMAL EQUATION\n');
fprintf('------------------------------------------------------------\n\n');

data = load('../data/data2.txt');

% ==============================================================================

% Select train and test dataset
X = data(:, 1:2);
y = data(:, 3);

% Number of observations
m = length(y);

% Treat the intercept term as another feature
X = [ones(m, 1) X];

% ==============================================================================

% Theta initialization
theta = normalEquation(X, y);

fprintf('Theta computed from the normal equations:\n%f\n%f\n%f\n\n', theta(1),theta(2), theta(3));

% ==============================================================================

fprintf('Cost after normal equation : %0.2f\n\n', meanSquaredErrorCost(X, y, theta));

% ==============================================================================

predict4 = [1 1650 3]*theta;

fprintf('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n$%f\n\n', predict4);
