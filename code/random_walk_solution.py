'''
 Let's write code that plots a 2D random walk
 given an input probability distribution:
 
 p[0] = probability of going north (positive y-axis)
 p[1] = probability of going east (positive x-axis)
 p[2] = probability of going south (negative y-axis)
 p[3] = probability of going west (negative x-axis)
        (all elements are at least 0)
        (all elements sum to 1)        
        
 Example: p = [0.4,0.25,0.1,0.25]

 The size of the step is random too!
 And drawn from a specified uniform distribution:
 u[0] = left edge of the uniform distribution
        (has to be larger than 0)
 u[1] = right edge of the uniform distribution
        (has to be at least u[0])
 
 Example: u = [1,10]
 
 The number of steps is specified as well: N.

 Start your walker at (x=0,y=0) 
 
 Functions you should look up and use:
 - np.random.multinomial() <- this has a funky output!
 - np.random.uniform() <- a more expected output
 - np.zeros() <- not required but a good idea
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster") # makes thicker lines

# example inputs to develop the code
p = [0.25,0.25,0.25,0.25]
u = [1,10]
N = 1000

# allocate the location arrays
locsX = np.zeros(N+1) # east and west
locsY = np.zeros(N+1) # north and south
#locsX = [0]*(N+1)
#locsY = [0]*(N+1)

# take the steps
for i in range(1,N+1):
    # first, pick a direction randomly
    direction = np.random.multinomial(n=1,pvals=p)  
    # next pick a step size randomly
    stepSize = np.random.gumbel()
    
    # now we move
    if direction[0] == 1: # if north
        locsY[i] = locsY[i-1] + stepSize
        locsX[i] = locsX[i-1]
    elif direction[1] == 1: # if east
        locsY[i] = locsY[i-1]
        locsX[i] = locsX[i-1] + stepSize
    elif direction[2] == 1: # if south
        locsY[i] = locsY[i-1] - stepSize
        locsX[i] = locsX[i-1]
    elif direction[3] == 1: # if west
        locsY[i] = locsY[i-1]
        locsX[i] = locsX[i-1] - stepSize  
        
# plot the result
plt.plot(locsX,locsY)
plt.xlabel('West to East')
plt.ylabel('South to North')
plt.show()