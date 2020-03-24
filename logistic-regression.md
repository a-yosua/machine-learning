# Logistic regression

This section will build a logistic regression model to predict the chance of student admission to the university based on their past two exams.

The MATLAB codes in this section is based on the exercises from Andrew Ng's [Machine Learning](https://www.coursera.org/learn/machine-learning) course on Coursera.

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

The ``plotData`` function is defined as below.

```Matlab
function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% Find Indices of Positive and Negative Examples
pos = find(y==1); neg = find(y == 0);
% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y','MarkerSize', 7);

% =========================================================================

hold off;

end
```

The data is visualised as in figure below.

<img src="https://github.com/a-yosua/machine-learning/blob/master/images/examScore.png" width="400">

## Implementation

### Sigmoid function

The logistic regression hypothesis is defined as 

![h_\theta(x)=g(\theta^TX)](https://render.githubusercontent.com/render/math?math=h_%5Ctheta(x)%3Dg(%5Ctheta%5ETX))

where ``g`` is the sigmoid function which is defined as 

![g(z)=\frac{1}{1+e^{-z}}](https://render.githubusercontent.com/render/math?math=g(z)%3D%5Cfrac%7B1%7D%7B1%2Be%5E%7B-z%7D%7D).

The code below will implement sigmoid function.

```Matlab
function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

g = 1./(1+exp(-z));

% =============================================================

end
```

Run the code above as below to check whether the function works properly. For large positive values of ``x``, the sigmoid function should return value close to 1. For large negative values of ``x``, the sigmoid function should return value close to 0. For ``x=0``, the sigmoid function should return 0.5.

```Matlab
sigmoid(0)
```

### Cost function

The cost function for logistic regression is:

![J(\theta)=\frac{1}{m}\sum_{i=1}^{m}\[-y^{(i)}\textrm{log}(h_\theta(x^{(i)}))-(1-y^{(i)})\textrm{log}(1-h_\theta(x^{(i)}))\]](https://render.githubusercontent.com/render/math?math=J(%5Ctheta)%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5B-y%5E%7B(i)%7D%5Ctextrm%7Blog%7D(h_%5Ctheta(x%5E%7B(i)%7D))-(1-y%5E%7B(i)%7D)%5Ctextrm%7Blog%7D(1-h_%5Ctheta(x%5E%7B(i)%7D))%5D)

### Cost gradient

The cost gradient with a length of ![\theta](https://render.githubusercontent.com/render/math?math=%5Ctheta) is defined as below:

![\frac{\partial J(\theta)}{\partial \theta_j}=\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20J(%5Ctheta)%7D%7B%5Cpartial%20%5Ctheta_j%7D%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D(h_%5Ctheta(x%5E%7B(i)%7D)-y%5E%7B(i)%7D)x%5E%7B(i)%7D_j)

The code below will implement the cost function and gradient in ``costFunction``.

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

The code below calls ``costFunction`` with initial parameters of ![\theta](https://render.githubusercontent.com/render/math?math=%5Ctheta).

```Matlab
% Setup the data matrix appropriately
[m, n] = size(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Initialize the fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display the initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);
fprintf('Cost at initial theta (zeros): %f\n', cost);
```

Output:
```
Cost at initial theta (zeros): 0.693147
```

```Matlab
disp('Gradient at initial theta (zeros):'); disp(grad);
```

Output:
```
Gradient at initial theta (zeros):
   -0.1000
  -12.0092
  -11.2628
```

### Matlab's fminunc

Finding the best parameters ![\theta](https://render.githubusercontent.com/render/math?math=%5Ctheta) for the logistic regression cost function can also be done by calling ``fminunc`` function given a fixed dataset of ``X`` and ``y`` values. Since ![\theta](https://render.githubusercontent.com/render/math?math=%5Ctheta) in logistic regression does not have constraints to take any real value, ``fminunc`` function can be used to finding minimum of unconstrained multivariable. Contraints in optimization refers to constraints on the parameters bound the possible ![\theta](https://render.githubusercontent.com/render/math?math=%5Ctheta) can take.

1. Set the ``GradObj`` option to ``on`` so that ``fminunc`` function returns the cost and the gradient. This option makes ``fminunc`` function to use the gradient when minimizing the function.
2. Set the ``MaxIter`` option to 400 so that ``fminunc`` function runs for 400 steps.
3. Specify the function to be minimized by using ``@(t)(costFunction(t,X,y))`` with argument ``t`` which calls the ``costFunction``  function defined previously.

After setting up the ``fminunc`` correctly, it should converge to the optimised parameters. The ``fminunc`` function will return the final cost and ![\theta](https://render.githubusercontent.com/render/math?math=%5Ctheta). The final ![\theta](https://render.githubusercontent.com/render/math?math=%5Ctheta) will be used to plot the decision boundary on the training data by ``plotDecisionBoundary`` function.

```Matlab
% Set options for fminunc
options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 400);

% Run fminunc to obtain the optimal theta
% This function will return theta and the cost
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
```

Output:

```
Local minimum found.
Optimization completed because the size of the gradient is less than
the value of the optimality tolerance.
<stopping criteria details>
```

Check the cost found by ``fminunc``:

```Matlab
% Print theta
fprintf('Cost at theta found by fminunc: %f\n', cost);
```

Output:
```
Cost at theta found by fminunc: 0.203498
```

Check the ![\theta](https://render.githubusercontent.com/render/math?math=%5Ctheta):


```Matlab
disp('theta:');disp(theta);
```

Output:
```
theta:
  -25.1613
    0.2062
    0.2015
```

Plot the decision boundary based on the calculated ![\theta](https://render.githubusercontent.com/render/math?math=%5Ctheta):


```Matlab
% Plot Boundary
plotDecisionBoundary(theta, X, y);

% Add some labels
hold on;

% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;
```

Output: 

<img src="https://github.com/a-yosua/machine-learning/blob/master/images/decisionBoundary.png" width="400">
