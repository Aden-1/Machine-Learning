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

#import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load the data from Housing_data.txt
data_loc = "Housing_data.txt"
def load_data():
    csv_path = data_loc
    return pd.read_csv(csv_path, sep=',', low_memory=False, names=["size", "room_count", "price"], dtype=float)

#Set data to the Pandas dataframe with the housing information
data = load_data()

####################################################################

# Print out some data points

#Get the basic info
print(data.info())

#Get a desciription of the data
print(data.describe())

#Get the first 5 rows
print(data.head())

#Get the last 5 rows
print(data.tail())


####################################################################


# Scale features and set them to zero mean
# ====================== YOUR CODE HERE ======================
#               First, for each feature dimension, compute the mean
#               of the feature and subtract it from the dataset.  
#               Next, compute the standard deviation of each feature and divide
#               Next, You need to perform the normalization separately for each feature.


#Create a pd dataframe to hold the mean and std for each feature
Summaries = pd.DataFrame(columns=["size", "room_count"])

#Scales the data.
#for every feature in the df get the mean & std then for every observation subtract it from the mean and then divide by the std
for feature in data:
    # Do not scale the target, only the features
    if feature != "price":
        #Get mean and std for the feature
        mean = data[feature].mean()
        std = data[feature].std()
        #For every item in that feature apply the standardization
        data[feature] = (data[feature] - mean) / std

        # Save summaries for future
        Summaries.loc["mean", feature] = mean
        Summaries.loc["std", feature] = std

####################################################################



# Print out some data points
# ====================== YOUR CODE HERE ======================

#After applying the standardization check the data again.

#Get the basic info
print(data.info())

#Get a desciription of the data
print(data.describe())

#Get the first 5 rows
print(data.head())

#Get the last 5 rows
print(data.tail())
####################################################################

## =================== Part 2: Cost and Gradient descent ===================

# ====================== YOUR CODE HERE ======================

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# Init Theta and Run Gradient Descent

#Randomly initialize 3 theta values.
theta = np.random.rand(3,1) # random initialization

#Set Y to the target price
Y = pd.DataFrame(data["price"])

#Set X to the input features size and room_count while adding a row of ones at the beginning for theta0
X_b = np.c_[np.ones((len(data), 1)), data["size"], data["room_count"]]  # add x0 = 1 to each instance

#for every iteration of gradient descent
for iter in range(iterations):
    #gen the gradient
    gradients = 2/len(data) * X_b.T.dot(X_b.dot(theta) - Y)
    #set theta to a new value by getting the difference of the current theta and the alpha * the gradient
    theta = theta - alpha * gradients
    #for every 300 iterations print the current theta value
    if iter % 300 == 0:
        print("ROUND", iter)
        print(theta)


# Perform a single gradient step on the parameter thetas.

# 1 Step gradient
#MAY NO BE NEEDED
#Get a new random theta
theta2 = np.random.rand(3,1) # random initialization
# Apply 1 step of gradient descent
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
#Get an array of 1-len of costs for the plot
arr = np.arange(len(Costs))
#Create the plot
plt.plot(arr, Costs)
#Show the plot
plt.show()

####################################################################

# Display gradient descent's result
# ====================== YOUR CODE HERE ======================

# Print the final theta values
print("Final theta:\n",theta)

####################################################################

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================

#Function to predict the value of a home wit the sqf and rooms
def predict(sqf, rooms):

    #Get the scaled sqf and rooms using stored mean and std
    temp_sqf = (sqf - Summaries.loc["mean", "size"]) / Summaries.loc["std", "size"]
    temp_rooms = (rooms - Summaries.loc["mean", "room_count"]) / Summaries.loc["std", "room_count"]

    #Set the X value to [1, sqf, temp_rooms]
    X_new = (1, temp_sqf, temp_rooms)
    #get the price using the learned linear regression values on X_new
    price_estimate = theta.T.dot(X_new)
    #return the estimate
    return price_estimate

#Print the estimated value of the house
print("ESTIMATE:", predict(1650, 3))

####################################################################

#Verify that our learned theta values make sense.

#import sklearn
from sklearn.linear_model import LinearRegression

#Set X to the size and room_count
X = data[["size", "room_count"]]

#Apply linear regression and fit it to the features and the target
lin_reg = LinearRegression()
lin_reg.fit(X, Y)
#print the theta values
print(lin_reg.intercept_, lin_reg.coef_)