function [theta, J_history] = linearRegressionGradientDescent(X, y, theta, alpha, num_iters)
  %LINEARREGRESSIONGRADIENTDESCENT Performs gradient descent to learn theta
  %   theta = LINEARREGRESSIONGRADIENTDESCENT(x, y, theta, alpha, num_iters)
  %   updates theta by taking num_iters gradient steps with learning rate alpha

  % Initialize some useful values
  m = length(y); % number of training examples
  J_history = zeros(num_iters, 1);

  for iter = 1:num_iters
      h_x = X*theta;

      theta = theta - alpha * (1 / m) * X' * (X*theta - y);
      % Save the cost J in every iteration
      J_history(iter) = meanSquaredErrorCost(X, y, theta);
  end
end
