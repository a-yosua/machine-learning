# Regularized logistic regression

This section is based on the previous article about [logistic regression](../master/logistic-regression.md). Regularization addresses overfitting by keeping all features but reducing the value of parameters ![\theta](https://render.githubusercontent.com/render/math?math=%5Ctheta).

The training data is from the past test results of the microchips containing two different test scores. We want to build a logistic regression model based on this training data and predict whether the microchip will pass the test (accept or reject) based on the two different test scores. The MATLAB codes are based on the exercises from Andrew Ng's [Machine Learning](https://www.coursera.org/learn/machine-learning) Week 3 course on Coursera.

## Data overview

Let us visualize the training data consisting of test scores of the microchips:

```Matlab
%  The first two columns contains the X values and the third column
%  contains the label (y).
data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

plotData(X, y);
% Put some labels 
hold on;
% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
% Specified in plot order
legend('y = 1', 'y = 0')
hold off;
```

Output:

<img src="https://github.com/a-yosua/machine-learning/blob/master/images/regularizedLogisticRegression/qualityTest.png" width="400">

## Implementation

### Feature mapping

The dataset clearly shows that a straight-line cannot separate the accepted and rejected data points. More features from each data point should be introduced to fit the data better.

The ``mapFeature`` function will map the features to polynomial terms of ![x_1](https://render.githubusercontent.com/render/math?math=x_1) and ![x_2](https://render.githubusercontent.com/render/math?math=x_2) up to the sixth power:

![\textrm{mapFeature}(x)=\begin{bmatrix} 1 \\  x_1 \\  x_2 \\  x^2_1 \\  x^2_2 \\  x_1 x_2 \\ \vdots  \\  x^6_2 \end{bmatrix}](https://render.githubusercontent.com/render/math?math=%5Ctextrm%7BmapFeature%7D(x)%3D%5Cbegin%7Bbmatrix%7D%201%20%5C%5C%20%20x_1%20%5C%5C%20%20x_2%20%5C%5C%20%20x%5E2_1%20%5C%5C%20%20x%5E2_2%20%5C%5C%20%20x_1%20x_2%20%5C%5C%20%5Cvdots%20%20%5C%5C%20%20x%5E6_2%20%5Cend%7Bbmatrix%7D).

```Matlab

function out = mapFeature(X1, X2)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%

degree = 6;
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end

```

Map the features:

```
% Add Polynomial Features
% Note that mapFeature also adds a column of ones for us, so the intercept term is handled
X = mapFeature(X(:,1), X(:,2));
```

### Cost function

The cost function for regularized logistic regression is:

![J(\theta)=\frac{1}{m}\sum_{i=1}^{m}\[-y^{(i)}\textrm{log}(h_\theta(x^{(i)}))-(1-y^{(i)})\textrm{log}(1-h_\theta(x^{(i)}))\]+\frac{\lambda}{2m}\sum_{j=1}^{n}\theta^2_j](https://render.githubusercontent.com/render/math?math=J(%5Ctheta)%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5B-y%5E%7B(i)%7D%5Ctextrm%7Blog%7D(h_%5Ctheta(x%5E%7B(i)%7D))-(1-y%5E%7B(i)%7D)%5Ctextrm%7Blog%7D(1-h_%5Ctheta(x%5E%7B(i)%7D))%5D%2B%5Cfrac%7B%5Clambda%7D%7B2m%7D%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%5Ctheta%5E2_j)

### Cost gradient

The cost gradient is defined as below:

![\frac{\partial J(\theta)}{\partial \theta_j}=\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j \textrm{ for } j=0](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20J(%5Ctheta)%7D%7B%5Cpartial%20%5Ctheta_j%7D%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D(h_%5Ctheta(x%5E%7B(i)%7D)-y%5E%7B(i)%7D)x%5E%7B(i)%7D_j%20%5Ctextrm%7B%20for%20%7D%20j%3D0)

![\frac{\partial J(\theta)}{\partial \theta_j}=\left (\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j\right )+ \frac{\lambda}{m} \theta_j \textrm{ for } j=\geq 0](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20J(%5Ctheta)%7D%7B%5Cpartial%20%5Ctheta_j%7D%3D%5Cleft%20(%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D(h_%5Ctheta(x%5E%7B(i)%7D)-y%5E%7B(i)%7D)x%5E%7B(i)%7D_j%5Cright%20)%2B%20%5Cfrac%7B%5Clambda%7D%7Bm%7D%20%5Ctheta_j%20%5Ctextrm%7B%20for%20%7D%20j%3D%5Cgeq%200)

The parameter ![\theta_0](https://render.githubusercontent.com/render/math?math=%5Ctheta_0) should not be regularized.

The ``cosfFunctionReg`` function computes cost and gradient for regularized logistic regression:

```Matlab

function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

hypothesis = sigmoid(theta'*X'); % 1 x m matrix
cost = -y.*log(hypothesis') - (1-y).*log(1-hypothesis');

thetaReg = theta(2:size(theta,1),1); % exclude theta0
costReg = lambda/(2*m) * sum(thetaReg.^2);

J = 1/m * sum(cost) + costReg;

cost = hypothesis'-y; % m x 1 matrix
gradCost = cost.*X; % m x n+1 matrix
grad = 1/m .* sum(gradCost)' + lambda/m*theta; % n+1 x 1 matrix

grad(1) = 1/m .* sum(gradCost(:,1))'; % should not regularize theta(1) or j0

% =============================================================

end

```

The code below calls ``costFunctionReg`` function to get the cost and gradient value for regularized logistic regression using the initial value of ![\theta](https://render.githubusercontent.com/render/math?math=%5Ctheta) set to all zeros:

```Matlab

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);

```

Output:

```
Cost at initial theta (zeros): 0.693147
```

# Decision boundary

We use the ``fminunc`` to learn the optimal parameters. To plot the nonlinear decision boundary, we use ``plotDecisionBoundary`` available from Andrew Ng's [Machine Learning](https://www.coursera.org/learn/machine-learning) Week 3 course on Coursera.

The decision boundary changes when we vary the ![\lambda](https://render.githubusercontent.com/render/math?math=%5Clambda).

![\lambda=0](https://render.githubusercontent.com/render/math?math=%5Clambda%3D0)

<img src="https://github.com/a-yosua/machine-learning/blob/master/images/regularizedLogisticRegression/regularizedLogLambda0.png" width="400">

![\lambda=1](https://render.githubusercontent.com/render/math?math=%5Clambda%3D1)

<img src="https://github.com/a-yosua/machine-learning/blob/master/images/regularizedLogisticRegression/regularizedLogLambda1.png" width="400">

![\lambda=10](https://render.githubusercontent.com/render/math?math=%5Clambda%3D10)

<img src="https://github.com/a-yosua/machine-learning/blob/master/images/regularizedLogisticRegression/regularizedLogLambda10.png" width="400">

![\lambda=100](https://render.githubusercontent.com/render/math?math=%5Clambda%3D100)

<img src="https://github.com/a-yosua/machine-learning/blob/master/images/regularizedLogisticRegression/regularizedLogLambda100.png" width="400">

The training set accuracy also varies:
1. ![\lambda=0](https://render.githubusercontent.com/render/math?math=%5Clambda%3D0), ``Train Accuracy: 88.983015``
2. ![\lambda=1](https://render.githubusercontent.com/render/math?math=%5Clambda%3D1), ``Train Accuracy: 83.050847``
3. ![\lambda=10](https://render.githubusercontent.com/render/math?math=%5Clambda%3D10), ``Train Accuracy: 74.576271``
4. ![\lambda=100](https://render.githubusercontent.com/render/math?math=%5Clambda%3D100), ``Train Accuracy: 61.016949``

Setting the ![\lambda](https://render.githubusercontent.com/render/math?math=%5Clambda) to a large value might results in underfitting.
