# linear Regression with multivariables

import numpy as np
import matplotlib.pyplot as plt

datafile = 'data/ex1data2.txt'
# Read into the data file
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True) #Read in comma separated data
# Form the usual "X" matrix and "y" vector
X = np.transpose(np.array(cols[:-1])) # m rows n cols
y = np.transpose(np.array(cols[-1:])) # m rows
m = y.size # number of training examples
# Insert the usual column of 1's into the "X" matrix
X = np.insert(X,0,1,axis=1)

# Quick visualize data
plt.grid(True)
plt.xlim([-100,5000])
dummy = plt.hist(X[:,0],label = 'col1')
dummy = plt.hist(X[:,1],label = 'col2')
dummy = plt.hist(X[:,2],label = 'col3')
plt.title('Clearly we need feature normalization.')
plt.xlabel('Column Value')
plt.ylabel('Counts')
dummy = plt.legend()
plt.show()



# Feature normalizing the columns (subtract mean, divide by standard deviation) feature scaling
# Store the mean and std for later use
# Note don't modify the original X matrix, use a copy
stored_feature_means, stored_feature_stds = [], []
Xnorm = X.copy()
for icol in xrange(Xnorm.shape[1]):
    stored_feature_means.append(np.mean(Xnorm[:,icol]))
    stored_feature_stds.append(np.std(Xnorm[:,icol]))
    #Skip the first column
    if icol == 0: continue
    #Faster to not recompute the mean and std again, just used stored values
    Xnorm[:,icol] = (Xnorm[:,icol] - stored_feature_means[-1])/stored_feature_stds[-1]



#Quick visualize the feature-normalized data
plt.grid(True)
plt.xlim([-5,5])
dummy = plt.hist(Xnorm[:,0],label = 'col1')
dummy = plt.hist(Xnorm[:,1],label = 'col2')
dummy = plt.hist(Xnorm[:,2],label = 'col3')
plt.title('Feature Normalization Accomplished')
plt.xlabel('Column Value')
plt.ylabel('Counts')
dummy = plt.legend()
plt.show()


# Gradient Descent
iterations = 1500
alpha = 0.01


def h(theta,X): # Linear hypothesis function

    return np.dot(X, theta)


def compute_cost(mytheta, X, y): # Cost function
    """
    theta_start is an n- dimensional vector of initial theta guess
    X is matrix with n- columns and m- rows
    y is a matrix with m- rows and 1 column
    """
    # note to self: *.shape is (rows, columns)
    # inner = np.power((h(mytheta, X) - y), 2)
    # return np.sum(inner) / (2 * m)

    return float((1/(2*m)) * np.dot((h(mytheta,X)-y).T, (h(mytheta,X)-y)))


# Actual gradient descent minimizing routine
def descend_gradient(X, theta_start = np.zeros(2)):
    """
    theta_start is an n- dimensional vector of initial theta guess
    X is matrix with n- columns and m- rows
    """
    theta = theta_start
    jvec = [] # Used to plot cost as function of iteration
    thetahistory = [] # Used to visualize the minimization path later on
    for meaninglessvariable in xrange(iterations):
        tmptheta = theta
        jvec.append(compute_cost(theta,X,y))
        thetahistory.append(list(theta[:,0]))
        # Simultaneously updating theta values
        for j in xrange(len(tmptheta)):
            # tmptheta[j] = theta[j] - (alpha/m)*np.sum((h(initial_theta,X) - y)*np.array(X[:,j]).reshape(m,1))
            tmptheta[j] = theta[j] - (alpha / m) * np.dot((h(theta, X) - y).T, np.array(X[:, j]).reshape(m, 1))
        theta = tmptheta
        # theta = theta - (alpha/m)*np.sum(h(initial_theta, X)*X)
    return theta, thetahistory, jvec


# Plot the convergence of the cost function
def plot_convergence(jvec):
    plt.figure(figsize=(10,6))
    plt.plot(range(len(jvec)),jvec,'bo')
    plt.grid(True)
    plt.title("Convergence of Cost Function")
    plt.xlabel("Iteration number")
    plt.ylabel("Cost function")
    plt.xlim([-0.05*iterations,1.05*iterations])
    # plt.ylim([min(jvec),max(jvec)])


# Run gradient descent with multiple variables, initial theta still set to zeros
# (Note! This doesn't work unless we feature normalize! "overflow encountered in multiply")
initial_theta = np.zeros((Xnorm.shape[1],1))
theta, thetahistory, jvec = descend_gradient(Xnorm,initial_theta)

# Plot convergence of cost function:
# plot_convergence(jvec)
# plt.show()


# print "Final result theta parameters: \n",theta
print "Check of result: What is price of house with 1650 square feet and 3 bedrooms?"
ytest = np.array([1650.,3.])
# To "undo" feature normalization, we "undo" 1650 and 3, then plug it into our hypothesis
ytestscaled = [(ytest[x]-stored_feature_means[x+1])/stored_feature_stds[x+1] for x in xrange(len(ytest))]
ytestscaled.insert(0,1)
print "$%0.2f" % float(h(theta,ytestscaled))


from numpy.linalg import inv
# Implementation of normal equation to find analytic solution to linear regression
def normEqtn(X,y):
    #restheta = np.zeros((X.shape[1],1))
    return np.dot(np.dot(inv(np.dot(X.T,X)),X.T),y)



print "Normal equation prediction for price of house with 1650 square feet and 3 bedrooms"
print "$%0.2f" % float(h(normEqtn(X,y),[1,1650.,3]))