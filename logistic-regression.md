# Logistic regression

This section will build a logistic regression model to predict the chance of student admission to the university based on their past two exams.

This MATLAB codes in this section is based on the exercises from Andrew Ng's [Machine Learning](https://www.coursera.org/learn/machine-learning) course on Coursera.

## Data overview

The code below will visualise the training data.

```Matlab
% Plot the data with + indicating (y = 1) examples and o indicating (y = 0) examples.
plotData(X, y);
 
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
```

## Cost function

The code below will implement the cost function.

```Matlab
function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

hypothesis = sigmoid(theta'*X'); % 1 x m matrix
cost = -y.*log(hypothesis') - (1-y).*log(1-hypothesis');
J = 1/m * sum(cost);

cost = hypothesis'-y; % m x 1 matrix
gradCost = cost.*X;
grad = 1/m .* sum(gradCost)'; % n+1 x 1 matrix

% =============================================================

end
```
