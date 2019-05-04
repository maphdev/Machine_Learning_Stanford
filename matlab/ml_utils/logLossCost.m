function [J, grad] = logLossCost(X, y, theta)
  %LOGLOSSCOST Compute cost and gradient for logistic regression
  %   J = LOGLOSSCOST(theta, X, y) computes the cost of using theta as the
  %   parameter for logistic regression and the gradient of the cost
  %   w.r.t. to the parameters.

  % Initialize some useful values
  m = length(y); % number of training examples

  % Compute h_x
  z = X*theta;
  h_x = sigmoid(z);

  % Cost function and gradient
  J = (1/m)*sum(-y.*log(h_x)-(1-y).*log(1 - h_x));
  grad = (1/m)*X'*(h_x-y);
end
