function J = meanSquaredErrorCost(X, y, theta)
  %MEANSQUAREDERRORCOST Compute cost for linear regression
  %   J = MEANSQUAREDERRORCOST(X, y, theta) computes the cost of using
  %   theta as the parameter for linear regression to fit the data points in X
  %   and y

  % Initialize some useful values
  m = length(y); % number of training examples

  J = (1/(2*m))*sum((X*theta - y).^2);
  %J = (1/(2*m))*sum((theta(1, 1) + theta(2, 1)*X(:, 2) - y).^2);
  %J = (1/(2*m))*ones(1, m)*(X*theta - y).^2;
end
