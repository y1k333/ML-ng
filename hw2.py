# Logistic Regression


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


datafile = 'data/ex2data1.txt'
# !head $datafile
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True) # Read in comma separated data
# Form the usual "X" matrix and "y" vector
X = np.transpose(np.array(cols[:-1])) # m * n
y = np.transpose(np.array(cols[-1:])) # m * 1
m = y.size # number of training examples
# Insert the usual column of 1's into the "X" matrix
X = np.insert(X,0,1,axis=1)

# visualizing the data
# Divide the sample into two: ones with positive classification, one with null classification
pos = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 1])
neg = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 0])
# Check to make sure I included all entries
# print "Included everything? ",(len(pos)+len(neg) == X.shape[0])


def plot_data():
    # plt.figure(figsize=(10, 6))
    plt.plot(pos[:, 1], pos[:, 2], 'k+', label='Admitted')
    plt.plot(neg[:, 1], neg[:, 2], 'yo', label='Not admitted')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title("visualize the data")
    plt.legend()
    plt.grid(True)


plot_data()
plt.show()


# Implementation
from scipy.special import expit #Vectorized sigmoid function
# Quick check the sigmoid function
myx = np.arange(-10,10,.1)
plt.plot(myx,expit(myx))
plt.title("sigmoid function")
plt.grid(True)
plt.show()

# Hypothesis function and cost function for logistic regression
def h(mytheta, myX): # Logistic hypothesis function

    return expit(np.dot(myX,mytheta))



# cost funtion, default lambda (regularization) 0
def compute_cost(mytheta, myX, myy, mylambda = 0.):
    """
    mytheta is an n- dimensional vector of initial theta guess
    X is matrix with m- rows and n- columns
    y is a matrix with m- rows and 1 column
    Note this includes regularization, if you set mylambda to nonzero
    For the first part of the homework, the default 0. is used for mylambda
    """
    term1 = np.dot(np.array(myy).T, np.log(h(mytheta, myX)))
    term2 = np.dot((1 - np.array(myy)).T, np.log(1 - h(mytheta, myX)))
    regterm = (mylambda / 2) * np.sum(np.dot(mytheta[1:].T, mytheta[1:]))  # Skip theta0
    return float(-(1./m) * (term1 + term2 + regterm))


# Check that with theta as zeros, cost returns about 0.693:
initial_theta = np.zeros((X.shape[1],1))
print compute_cost(initial_theta,X,y)

# gradient function some problems
def gradient(mytheta, myX, myy):

    grad = np.zeros(len(mytheta))
    for j in xrange(len(grad)):
        grad[j] = (1 / m) * np.sum((h(mytheta, myX) - myy) * np.array(myX[:, j]))

    return grad


# An alternative to OCTAVE's 'fminunc' we'll use some scipy.optimize function, "fmin"
# Note "fmin" does not need to be told explicitly the derivative terms
# It only needs the cost function, and it minimizes with the "downhill simplex algorithm."
# http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.optimize.fmin.html
from scipy import optimize

def optimizeTheta(mytheta,myX,myy,mylambda=0.):
    result = optimize.fmin(compute_cost, x0=mytheta, args=(myX, myy, mylambda), maxiter=400, full_output=True)
    return result[0], result[1]

# some problems
def optimizeTheta1(mytheta,myX,myy,mylambda=0.):
    result = optimize.fmin_tnc(func=compute_cost, x0=mytheta, fprime=gradient, args=(myX, myy))
    return result[0], result[1]


theta, mincost = optimizeTheta(initial_theta,X,y)
# some problems
theta1, mincost1 = optimizeTheta1(initial_theta,X,y)
# That's pretty cool. Black boxes ftw


# Call your costFunction function using the optimal parameters of theta.
# You should see that the cost is about 0.203."
print compute_cost(theta,X,y)
print compute_cost(theta1,X,y)



# Plotting the decision boundary: two points, draw a line between
# Decision boundary occurs when h = 0, or when
# theta0 + theta1*x1 + theta2*x2 = 0
# y=mx+b is replaced by x2 = (-1/thetheta2)(theta0 + theta1*x1)

boundary_xs = np.array([np.min(X[:,1]), np.max(X[:,1])])
boundary_ys = (-1./theta[2])*(theta[0] + theta[1]*boundary_xs)
plot_data()
plt.plot(boundary_xs,boundary_ys,'b-',label='Decision Boundary')
plt.legend()
plt.show()


# For a student with an Exam 1 score of 45 and an Exam 2 score of 85,
# you should expect to see an admission probability of 0.776.
print h(theta,np.array([1, 45.,85.]))


def makePrediction(mytheta, myx):
    return h(mytheta,myx) >= 0.5

# Compute the percentage of samples I got correct:
pos_correct = float(np.sum(makePrediction(theta,pos)))
neg_correct = float(np.sum(np.invert(makePrediction(theta,neg))))
tot = len(pos)+len(neg)
prcnt_correct = float(pos_correct+neg_correct)/tot
print "Fraction of training samples correctly predicted: %f." % prcnt_correct
