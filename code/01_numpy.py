import numpy as np

N = 10000
# generate an array with N random numbers
x = np.random.random(N)
# add 1 to each element of x
x_plus_1 = x + 1

print(x)
print(x_plus_1)  


# doing the same thing with plain python lists
y = []
for i in range(N):
    y.append(np.random.random())

# add 1 to each element of y
y_plus_1 = []
for i in range(N):
    y_plus_1.append(y[i] + 1)
    
print(y)
print(y_plus_1)


# now we can use ipython to get times for both with %%timeit
# about the same speed, but numpy code is much more compact






# what else can we do with numpy arrays? lots!
x = np.random.randint(low=0,high=10,size=10)
print(x)
print(np.sin(x))
print(np.mean(x),np.median(x),np.std(x),np.var(x))
print(np.max(x),np.argmax(x))
print(np.maximum(0.9,x))
print(x > 0.5)
print(x[:N/2] / x[N/2:])
print(np.add.accumulate(x))
print(np.multiply.accumulate(x))




# make a times-table
x = np.arange(1, 10) # kind of like range(1,10), but a numpy array
print(np.multiply.outer(x, x))




# aggregating over matrices
x = np.random.rand(3, 5) # array of arrays
print(x)
print(np.mean(x))
print(np.mean(x,axis=0))
print(np.mean(x,axis=1))
# we can also use the method version:
print(x.mean(axis=0))
# somewhat confusing, as this doesn't work:
print(x.median(axis=0))
# but this does:
print(np.median(x,axis=0))






# mini-assignment: write a function that takes as input a matrix X
# and returns a matrix that is a Z-score normalized version of X:
# that is, its columns have zero mean and standard deviation 1

def normalize(X):
    # first, demean X
    # then, make sure each column has standard deviation 1
    return X_normalized
    
# test code
X = np.random.randint(0,high=100,size=(10,3))
print(normalize(X))






# matrix algebra
x = np.random.randint(low=0,high=10,size=(2,3))

print(x + x**2) # elementwise squaring and adding

print(x,x.T,sep='\n\n') # transpose!

Y = np.dot(x,x.T) # matrix multiply
print(Y) 

Yinv = np.linalg.inv(Y) # more advanced linear algebra is in linalg
print(np.dot(Yinv,Y))



# broadcasting
print(x + 1) # adds 1 to every element
y = np.random.randint(low=0,high=10,size=(1,3))
print(x,y,x - y,sep='\n\n') # subtract y from each row of x!



# but this won't work because the sizes don't match up
y = np.random.randint(low=0,high=10,size=(3,1))
print(x,y,x - y,sep='\n\n') # subtract y from each row of x!




