% Path
addpath('ml_utils/')

% ==============================================================================

disp(sprintf('\n------------------------------------------------------------'));
disp(sprintf('LINEAR REGRESSION WITH ONE FEATURE'));
disp(sprintf('------------------------------------------------------------\n'));

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

disp(sprintf('Cost at initialization : %0.2f\n', computeMeanSquaredErrorCost(X, y, theta)));

% ==============================================================================

% Thetas with gradient descent
[theta, history] = gradientDescent(X, y, theta, alpha, num_iters);

% ==============================================================================

disp(sprintf('Theta computed from gradient descent:\n%f, \n%f\n',theta(1),theta(2)));

% ==============================================================================

disp(sprintf('Cost after gradient descent : %0.2f\n', computeMeanSquaredErrorCost(X, y, theta)));

% ==============================================================================

predict1 = [1, 3.5] *theta;
disp(sprintf('For population = 35,000, we predict a profit of %f\n', predict1*10000));

predict2 = [1, 7] * theta;
disp(sprintf('For population = 70,000, we predict a profit of %f\n', predict2*10000));

% ==============================================================================

disp(sprintf('------------------------------------------------------------'));
disp(sprintf('LINEAR REGRESSION WITH MULTIPLE FEATURE'));
disp(sprintf('------------------------------------------------------------\n'));

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

disp(sprintf('Cost at initialization : %0.2f\n', computeMeanSquaredErrorCost(X, y, theta)));

% ==============================================================================

% Thetas with gradient descent
[theta, hist] = gradientDescent(X, y, theta, alpha, num_iters);

% ==============================================================================

disp(sprintf('Theta computed from gradient descent:\n%f, \n%f, \n%f\n',theta(1),theta(2), theta(3)));

% ==============================================================================

disp(sprintf('Cost after gradient descent : %0.2f\n', computeMeanSquaredErrorCost(X, y, theta)));

% ==============================================================================

test = [1650 3];
normalizedTest = (test-mu)./sigma;
test = [1, normalizedTest];
predict3 = test*theta;

disp(sprintf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n$%f\n', predict3));

% ==============================================================================

disp(sprintf('------------------------------------------------------------'));
disp(sprintf('LINEAR REGRESSION WITH NORMAL EQUATION'));
disp(sprintf('------------------------------------------------------------\n'));

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

disp(sprintf('Theta computed from the normal equations:\n%f\n%f\n%f\n', theta(1),theta(2), theta(3)));

% ==============================================================================

disp(sprintf('Cost after normal equation : %0.2f\n', computeMeanSquaredErrorCost(X, y, theta)));

% ==============================================================================

predict4 = [1 1650 3]*theta;

disp(sprintf('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n$%f\n', predict4));
