import numpy as np
from sklearn import linear_model 
import matplotlib.pyplot as plt

from utilities import visualize_classifier

# LOGISTIC REGRESSION
# used to explain the relationship b/w input variables and output variables
# input variables = independednt
# output variables = dependednt variable
# The dependent variable (output) can take only a fixed set of values. These values
# correspond to the classes of the classification problem

# Identify the relationship b/w ind and dep variables by estimating the probabilitues using a logistic function
# This logistic function is SIGMOID CURVE thats used to build the function with various parameters
# Related to generalized linear model analysis, fit line to many points to minimize error

# Instead of linuear regression, we use logistic regression to facilitate classification

# Define sample input data
X = np.array([[3.1, 7.2], 
              [4, 6.7],
              [2.9, 8],
              [5.1, 4.5],
              [6, 5],
              [5.6, 5],
              [3.3, 0.4],
              [3.9, 0.9],
              [2.8, 1],
              [0.5, 3.4],
              [1, 4], 
              [0.6, 4.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
# categorize the data, first 3 points labeled with  0, then 3 labeled with 1, etc.
# sets backgroud color and groups points into sectinos  with logistic regression

# Create the logistic regression classifier
classifier = linear_model.LogisticRegression(solver='liblinear', C=100)

# Train the classifier
classifier.fit(X, y)

# Visualize the performance of the classifier 
visualize_classifier(classifier, X, y)
