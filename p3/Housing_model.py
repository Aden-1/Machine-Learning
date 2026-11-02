#you will implement linear regression with multiple variables to
#predict the prices of houses. Suppose you are selling your house and you
#want to know what a good market price would be. One way to do this is to
#first collect information on recent houses sold and make a model of housing
#prices.
#The file Housing_data.txt contains a training set of housing prices in Portland, Oregon. The first column is the size of the house (in square feet), the
#second column is the number of bedrooms, and the third column is the price of the house.
#
#  Instructions
#  ------------
# 
#  This file will give you outlines to get started on buiding a linear regression model. 
#

## ================ Part 1: Feature Normalization ================

## Load Housing_data.txt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_loc = "Housing_data.txt"
def load_data():
    csv_path = data_loc
    return pd.read_csv(csv_path, sep=',', low_memory=False, names=["size", "room_count", "price"], dtype=float)

data = load_data()

####################################################################

# Print out some data points

print(data.info())

print(data.describe())

print(data.head())

print(data.tail())


####################################################################


# Scale features and set them to zero mean
# ====================== YOUR CODE HERE ======================
#               First, for each feature dimension, compute the mean
#               of the feature and subtract it from the dataset.  
#               Next, compute the standard deviation of each feature and divide
#               Next, You need to perform the normalization separately for each feature.

Summaries = pd.DataFrame(columns=["size", "room_count"])

"""
      size    room_count
mean
std
"""


for feature in data:
    # Do not scale the target, only the features
    if feature != "price":
        mean = data[feature].mean()
        std = data[feature].std()
        data[feature] = (data[feature] - mean) / std

        # Save summaries for future
        Summaries.loc["mean", feature] = mean
        Summaries.loc["std", feature] = std

####################################################################



# Print out some data points
# ====================== YOUR CODE HERE ======================
print(data.info())

print(data.describe())

print(data.head())

print(data.tail())
####################################################################

## =================== Part 2: Cost and Gradient descent ===================

# ====================== YOUR CODE HERE ======================

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# Init Theta and Run Gradient Descent

#theta = np.random.rand(3,1) # random initialization

Y = pd.DataFrame(data["price"])

X_b = np.c_[np.ones((len(data), 1)), data["size"], data["room_count"]]  # add x0 = 1 to each instance

#for iter in range(iterations):
    #gradients = 2/len(data) * X_b.T.dot(X_b.dot(theta) - Y)
    #theta = theta - alpha * gradients
    #if iter % 300 == 0:
        #print("ROUND", iter)
        #print(theta)


# Perform a single gradient step on the parameter thetas.

# 1 Step gradient
theta2 = np.random.rand(3,1) # random initialization
theta2 = theta2 - alpha * 2 / len(data) * X_b.T.dot(X_b.dot(theta2) - Y)
	
#   After updating theta, compute the cost of using the current values of theta to fit the data points
#   You should set J_history(iter) to the cost of using the current values of theta.
#   Save the cost J in every iteration

###################################################################################


# Cost calculation
def J_history(theta):
    # Track error per row
    errors = []
    i = 0
    for i in range(len(X_b)):
        # Row i's prediction using current thetas
        pred = theta.T.dot(X_b[i])
        # Actual Y for row i
        actual = Y.iloc[i]
        # Error for row i prediction vs. actual
        e = (pred - actual) ** 2
        # Track all errors across rows
        errors.append(e)

    # Cost for current thetas
    cost = (1 / (len(X_b))) * sum(errors)
    return cost

theta = np.random.rand(3,1)
Costs = []
for iter in range(iterations):
    # Calculate gradient
    gradients = 2/len(data) * X_b.T.dot(X_b.dot(theta) - Y)
    # Update theta
    theta = theta - alpha * gradients

    # Get cost as a function of the current theta from J_history
    current_cost = J_history(theta)
    # Track costs across theta values
    Costs.append(current_cost)


# Plot the convergence graph
# ====================== YOUR CODE HERE ======================
arr = np.arange(len(Costs))
plt.plot(arr, Costs)
plt.show()

####################################################################

# Display gradient descent's result
# ====================== YOUR CODE HERE ======================

print("Final theta:\n",theta)

####################################################################

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================

def predict(sqf, rooms):

    temp_sqf = (sqf - Summaries.loc["mean", "size"]) / Summaries.loc["std", "size"]
    temp_rooms = (rooms - Summaries.loc["mean", "room_count"]) / Summaries.loc["std", "room_count"]

    X_new = (1, temp_sqf, temp_rooms)
    price_estimate = theta.T.dot(X_new)
    return price_estimate

print("ESTIMATE::::::::::::::::::", predict(1650, 3))

####################################################################

from sklearn.linear_model import LinearRegression

X = data[["size", "room_count"]]

lin_reg = LinearRegression()
lin_reg.fit(X, Y)
print(lin_reg.intercept_, lin_reg.coef_)