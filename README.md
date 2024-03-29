# Machine Learning Algorithms

This repository contains exercises implementing machine learning algorithms by using MATLAB.

[Logistic regression](../master/logistic-regression.md)

[Regularized logistic regression](../master/regularized-logistic-regression.md)

The following exercises build a neural network model by using TensorFlow.

[Simple neural network model](../master//TensorFlow_Exercise_1.ipynb)

[Learning different items in images](../master/TensorFlow_Exercise_2.ipynb)

# Machine Learning Deployment

From Django to call Amazon REST API to invoke Lambda function to predict cancer using the model endpoint deployed by SageMaker (http://34.227.49.154:8000/v1/api/predictbreastcancer/ - note: currently the endpoint is down).

The picture below shows how the model is deployed:

<img src="https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2021/03/31/1-diagram.jpg" width="500">

Test the API with Postman:

<img src="https://github.com/a-yosua/machine-learning/blob/master/images/amazon_api_test_1.png" width="600">

From Django to call Django REST API to return the prediction of type of flower from a pre-trained model with Scikit Learn (http://34.227.49.154:8000/v1/api/predictirisflower/).

Test the API with Postman:

<img src="https://github.com/a-yosua/machine-learning/blob/master/images/django_api_test_1.png" width="600">
