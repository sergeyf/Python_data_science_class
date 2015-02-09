# write a function that takes in a dataset
# and returns the index of each row's 
# nearest neighbor in euclidean distance
# (that is not itself)
# the output should be in a dictionary

# example input:
# X = np.array([[0,1,1],[1,1,0],[0,1,2]])
#
# example function call:
# getNN(X)
#
# example output:
# {0: 2, 1: 0, 2: 0}

# hint: make sure you understand euclidean distance
# hint: use broadcasting
# hint: use np.argsort

import numpy as np

# Euclidean distance function
def euclideanDist(X1,X2):
    # FILL STUFF HERE

# function that returns the nearest neighbors
def getNN(X):
    NN = {}
    for i,x in enumerate(X):
        # FILL STUFF HERE
        
    return NN

# testing
X = np.array([[0,1,1],[1,1,0],[0,1,2]])
print( getNN(X) )
    