function [theta, J_history] = logisticRegressionGradientDescent(X, y, theta, alpha, num_iters)
  %LOGISTICREGRESIONGRADIENTDESCENT Performs gradient descent to learn theta
  %   theta = LOGISTICREGRESIONGRADIENTDESCENT(x, y, theta, alpha, num_iters)
  %   updates theta by taking num_iters gradient steps with learning rate alpha

  % Initialize some useful values
  m = length(y); % number of training examples
  J_history = zeros(num_iters, 1);

  for iter = 1:num_iters
      z = X*theta;
      h_x = sigmoid(z);

      theta = theta - alpha * (1 / m) * X' * (h_x - y);
      % Save the cost J in every iteration
      J_history(iter) = logLossCost(X, y, theta);
  end
end
