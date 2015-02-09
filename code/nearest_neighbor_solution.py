# write a function that takes in a dataset
# and returns the index of each row's 
# nearest neighbor (in euclidean distance)
# (that isn't itself)
# the output should be in a dictionary

# example input:
# X = np.array([[0,1,1],[1,1,0],[0,1,2]])
#
# example function call:
# getNN(X)
#
# example output:
# {0: 2, 1: 0, 2: 0}

# hint: use broadcasting!
# hint: use np.argsort

import  numpy as np

# Euclidean distance function
def euclideanDist(X1,X2):
    return np.sum( (X1-X2)**2 , axis = 1)

# function that returns the nearest neighbors
def getNN(X):
    NN = {}
    for i,x in enumerate(X):
        dists = euclideanDist(X,x)
        NN[i] = np.argsort(dists)[1]
        
    return NN

# testing
X = np.array([[0,1,1],[1,1,0],[0,1,2]])
print( getNN(X) )
    