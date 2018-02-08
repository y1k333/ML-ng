# linear regression with one variable


import numpy as np
import matplotlib.pyplot as plt

# read data
datafile = 'data/ex1data1.txt'
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1),unpack=True) #Read in comma separated data
# Form the usual "X" matrix and "y" vector
X = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))

m = y.size # number of training examples

# Insert the usual column of 1's into the "X" matrix X0 = 1
X = np.insert(X,0,1,axis=1)

# Plot the data to see what it looks like
plt.plot(X[:,1],y[:,0],'rx',markersize=10)
plt.grid(True) #Always plot.grid true!
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')

# Gradient Descent
iterations = 1500
alpha = 0.01


def h(theta,X): # Linear hypothesis function

    return np.dot(X, theta) # diancheng


def compute_cost(mytheta, X, y): # Cost function
    """
    theta_start is an n- dimensional vector of initial theta guess
    X is matrix with n- columns and m- rows
    y is a matrix with m- rows and 1 column
    """
    # note to self: *.shape is (rows, columns)
    # inner = np.power((h(mytheta, X) - y), 2)
    # return np.sum(inner) / (2 * m)
    return float(np.dot(((h(mytheta,X)-y).T),(h(mytheta,X)-y)) / (2 * m))


# Test that running computeCost with 0's as theta returns 32.07:


initial_theta = np.zeros((X.shape[1],1)) # (theta is a vector with n rows and 1 columns (if X has n features) )


print compute_cost(initial_theta, X, y)


# Actual gradient descent minimizing routine
def descend_gradient(X, theta_start = np.zeros(2)):
    """
    theta_start is an n- dimensional vector of initial theta guess
    X is matrix with m- rows and n- columns
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
            # tmptheta[j] = theta[j] - (alpha/m)*np.sum((h(theta,X) - y)*np.array(X[:,j]).reshape(m,1))
            tmptheta[j] = theta[j] - (alpha / m) * np.dot((h(theta, X) - y).T, np.array(X[:, j]).reshape(m, 1))
        theta = tmptheta
        # theta = theta - (alpha/m)*np.sum(h(initial_theta, X)*X)
    return theta, thetahistory, jvec

# Actually run gradient descent to get the best-fit theta values
initial_theta = np.zeros((X.shape[1],1))
theta, thetahistory, jvec = descend_gradient(X,initial_theta)

# Plot the convergence of the cost function
def plot_convergence(jvec):
    plt.figure(figsize=(10,6))
    plt.plot(range(len(jvec)),jvec,'bo')
    plt.grid(True)
    plt.title("Convergence of Cost Function")
    plt.xlabel("Iteration number")
    plt.ylabel("Cost function")
    dummy = plt.xlim([-0.05*iterations,1.05*iterations])
    #dummy = plt.ylim([4,8])

plot_convergence(jvec)
dummy = plt.ylim([4,7])
plt.show()

#Plot the line on top of the data to ensure it looks correct
def myfit(xval):
    return theta[0] + theta[1]*xval


plt.figure(figsize=(10,6))
plt.plot(X[:,1],y[:,0],'rx',markersize=10,label='Training Data')
plt.plot(X[:,1],myfit(X[:,1]),'b-',label = 'Hypothesis: h(x) = %0.2f + %0.2fx'%(theta[0],theta[1]))
plt.grid(True) #Always plot.grid true!
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.legend()
plt.show()

# visualizling J(cita)
# Import necessary matplotlib tools for 3d plots

from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import itertools

fig = plt.figure(figsize=(12,12))
ax = fig.gca(projection='3d')

xvals = np.arange(-10,10,.5)
yvals = np.arange(-1,4,.1)
myxs, myys, myzs = [], [], []
for david in xvals:
    for kaleko in yvals:
        myxs.append(david)
        myys.append(kaleko)
        myzs.append(compute_cost(np.array([[david], [kaleko]]),X,y))

scat = ax.scatter(myxs,myys,myzs,c=np.abs(myzs),cmap=plt.get_cmap('YlOrRd'))

plt.xlabel(r'$\theta_0$',fontsize=30)
plt.ylabel(r'$\theta_1$',fontsize=30)
plt.title('Cost (Minimization Path Shown in Blue)',fontsize=30)
plt.plot([x[0] for x in thetahistory],[x[1] for x in thetahistory],jvec,'bo-')
plt.show()

