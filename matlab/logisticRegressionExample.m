% Path
addpath('ml_utils/')

% ==============================================================================

fprintf('\n------------------------------------------------------------\n');
fprintf('LOGISTIC REGRESSION\n');
fprintf('------------------------------------------------------------\n\n');

% Load the dataset
data = load('../data/data3.txt');

% ==============================================================================

% Select train and test dataset
X = data(:, 1:2);
y = data(:, 3);

%  Setup the data matrix appropriately
[m, n] = size(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Initialize the fitting parameters
initial_theta = zeros(n + 1, 1);

% ==============================================================================

% Compute the initial cost and gradient
[cost, grad] = logLossCost(X, y, initial_theta);

fprintf('Cost at initial theta (zeros): %f\n\n', cost);

% ==============================================================================

% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Run fminunc to obtain the optimal theta
[theta, cost] = fminunc(@(t)(logLossCost(X, y, t)), initial_theta, options);

% Display cost and theta
fprintf('Cost found by fminunc: %f\n\n', cost);
fprintf('Theta computed by fminunc:\n%f, \n%f, \n%f\n\n',theta(1),theta(2), theta(3));

% ==============================================================================

%  Predict probability for a student with score 45 on exam 1  and score 85 on exam 2
prob = sigmoid([1 45 85] * theta);
fprintf('For a student with scores 45 and 85, we predict an admission probability of %f\n\n', prob);

% Compute accuracy on our training set
p = predictLogisticRegresion(theta, X);
fprintf('Train Accuracy with fminunc: %f\n\n', mean(double(p == y)) * 100);

% ==============================================================================
fprintf('------------------------------------------------------------\n\n');

[theta, J_history] = logisticRegressionGradientDescent(X, y, initial_theta, 0.001, 300000);
fprintf('Cost found by gradient descent : %f\n\n', J_history(end));
fprintf('Theta computed by gradient descent :\n%f, \n%f, \n%f\n\n',theta(1),theta(2), theta(3));

% ==============================================================================

%  Predict probability for a student with score 45 on exam 1  and score 85 on exam 2
prob = sigmoid([1 45 85] * theta);
fprintf('For a student with scores 45 and 85, we predict an admission probability of %f\n\n', prob);

% Compute accuracy on our training set
p = predictLogisticRegresion(theta, X);
fprintf('Train Accuracy with gradient descent: %f\n', mean(double(p == y)) * 100);

% ==============================================================================

fprintf('\n------------------------------------------------------------\n');
fprintf('LOGISTIC REGRESSION WITH MAP FEATURES AND REGULARIZATION\n');
fprintf('------------------------------------------------------------\n\n');

% Load the dataset
data = load('../data/data4.txt');

% ==============================================================================

% Select train and test dataset
X = data(:, 1:2);
y = data(:, 3);

% Map the features into all polynomial terms of x& and x2 up to the sixth power
% Note that mapFeature also adds a column of ones for us, so the intercept term is handled
X = mapPolynomialFeature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% ==============================================================================

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic regression
[cost, grad] = logLossRegularizedCost(X, y, initial_theta, lambda);
fprintf('Cost at initial theta (zeros): %f\n\n', cost);

% ==============================================================================

% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Run fminunc to obtain the optimal theta
[theta, cost] = fminunc(@(t)(logLossRegularizedCost(X, y, t, lambda)), initial_theta, options);

% Display cost and theta
fprintf('Cost found by fminunc: %f\n\n', cost);
fprintf('Theta computed by fminunc:\n%f, \n%f, \n%f\n\n',theta(1),theta(2), theta(3));

% ==============================================================================

% Compute accuracy on our training set
p = predictLogisticRegresion(theta, X);
fprintf('Train Accuracy with gradient descent: %f\n\n', mean(double(p == y)) * 100);
