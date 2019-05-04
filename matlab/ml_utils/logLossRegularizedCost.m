function [J, grad] = logLossRegularizedCost(X, y, theta, lambda)
  %LOGLOSSREGULARIZEDCOST Compute cost and gradient for logistic regression with regularization
  %   J = LOGLOSSREGULARIZEDCOST(X, y, theta, lambda) computes the cost of using
  %   theta as the parameter for regularized logistic regression and the
  %   gradient of the cost w.r.t. to the parameters.

  m = length(y); % number of training examples

  grad = zeros(size(theta));

  z = X*theta;
  h_x = sigmoid(z);

  J = (1/m)*sum(-y.*log(h_x)-(1-y).*log(1 - h_x)) + (lambda/(2*m))*sum(theta(2:end).^2);

  grad(1) = (1/m)*X(:, 1)'*(h_x-y);
  grad(2:end) = (1/m)*X(:, 2:end)'*(h_x-y)+(lambda/m)*theta(2:end);
end
