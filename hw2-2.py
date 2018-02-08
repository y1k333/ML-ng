# Logistic Regression with regularization
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# visualize data
datafile = 'data/ex2data2.txt'
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True) # Read in comma separated data
# Form the usual "X" matrix and "y" vector
X = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))
m = y.size # number of training examples
# Insert the usual column of 1's into the "X" matrix
X = np.insert(X,0,1,axis=1)


# Divide the sample into two: ones with positive classification, one with null classification
pos = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 1])
neg = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 0])
# Check to make sure I included all entries
# print "Included everything? ",(len(pos)+len(neg) == X.shape[0])



def plot_data():
    plt.plot(pos[:,1],pos[:,2],'k+',label='y=1')
    plt.plot(neg[:,1],neg[:,2],'yo',label='y=0')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend()
    plt.grid(True)

# Draw it square to emphasize circular features
plt.figure(figsize=(6,6))
plot_data()

# feature mapping
def mapFeature(x1col, x2col):
    """
    Function that takes in a column of n- x1's, a column of n- x2s, and builds
    a n- x 28-dim matrix of features as described in the homework assignment
    """
    degrees = 6
    out = np.ones((x1col.shape[0], 1))

    for i in range(1, degrees+1):
        for j in range(0, i+1):
            term1 = x1col ** (i-j)
            term2 = x2col ** (j)
            term  = (term1 * term2).reshape( term1.shape[0], 1 )
            out   = np.hstack(( out, term ))
    return out


# Create feature-mapped X matrix
mappedX = mapFeature(X[:,1],X[:,2])

from scipy.special import expit #Vectorized sigmoid function
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

# cost function and gradient
# Cost function is the same as the one implemented above, as I included the regularization
# toggled off for default function call (lambda = 0)
# I do not need separate implementation of the derivative term of the cost function
# Because the scipy optimization function I'm using only needs the cost function itself
# Let's check that the cost function returns a cost of 0.693 with zeros for initial theta,
# and regularized x values
initial_theta = np.zeros((mappedX.shape[1],1))
compute_cost(initial_theta,mappedX,y)

# Learning parameters using fminunc
# I noticed that fmin wasn't converging (passing max # of iterations)
# so let's use minimize instead
from scipy import optimize

def optimizeRegularizedTheta(mytheta, myX, myy, mylambda=0.):
    result = optimize.minimize(compute_cost, mytheta, args=(myX, myy, mylambda), method='BFGS',
                               options={"maxiter": 500, "disp": False})
    return np.array([result.x]), result.fun


theta, mincost = optimizeRegularizedTheta(initial_theta, mappedX, y)



def plotBoundary(mytheta, myX, myy, mylambda=0.):
    """
    Function to plot the decision boundary for arbitrary theta, X, y, lambda value
    Inside of this function is feature mapping, and the minimization routine.
    It works by making a grid of x1 ("xvals") and x2 ("yvals") points,
    And for each, computing whether the hypothesis classifies that point as
    True or False. Then, a contour is drawn with a built-in pyplot function.
    """
    theta, mincost = optimizeRegularizedTheta(mytheta,myX,myy,mylambda)
    xvals = np.linspace(-1,1.5,50)
    yvals = np.linspace(-1,1.5,50)
    zvals = np.zeros((len(xvals),len(yvals)))
    for i in xrange(len(xvals)):
        for j in xrange(len(yvals)):
            myfeaturesij = mapFeature(np.array([xvals[i]]),np.array([yvals[j]]))
            zvals[i][j] = np.dot(theta,myfeaturesij.T)
    zvals = zvals.transpose()

    u, v = np.meshgrid( xvals, yvals )
    mycontour = plt.contour( xvals, yvals, zvals, [0])
    #Kind of a hacky way to display a text on top of the decision boundary
    myfmt = { 0:'Lambda = %d'%mylambda}
    plt.clabel(mycontour, inline=1, fontsize=15, fmt=myfmt)
    plt.title("Decision Boundary")

# Build a figure showing contours for various values of regularization parameter, lambda
# It shows for lambda=0 we are overfitting, and for lambda=100 we are underfitting
plt.figure(figsize=(12, 10))
plt.subplot(221)
plot_data()
plotBoundary(theta, mappedX, y, 0.)

plt.subplot(222)
plot_data()
plotBoundary(theta, mappedX, y, 1.)

plt.subplot(223)
plot_data()
plotBoundary(theta, mappedX, y, 10.)

plt.subplot(224)
plot_data()
plotBoundary(theta, mappedX, y, 100.)

plt.show()