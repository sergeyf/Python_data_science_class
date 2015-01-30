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







# what else can we do with numpy arrays? lots!
N = 10
x = np.random.randint(low=0,high=10,size=N)
print(x)

print(np.sin(x)) # take sin of every element

print(np.mean(x),np.median(x),np.var(x)) # summary statistics

print(np.max(x),np.argmax(x)) # x[np.argmax(x)] == np.max(x)

print(np.maximum(5,x)) # elementwise maximum

y = np.random.randint(low=0,high=10,size=N)

print('x=',x,'y=',y,'max of each element:',np.maximum(x,y),sep='\n')

print(x > 5) # elementwise boolean comparisons against 5

print(x > y) # ditto, but x[i] is compared y[i] for each i

print(x[:N/2] / x[N/2:]) # first half of x divided by the second half

print(np.add.accumulate(x)) # cumulative sum, same as np.cumsum()

print(np.multiply.accumulate(x)) # cumulative product, same as np.cumprod()





# more examples
x = np.arange(1, 10) # like range(1,10), but a numpy array
print(np.multiply.outer(x, x)) 






# aggregating over matrices
X = np.random.rand(3, 5) # array of arrays
print(X)
print(np.mean(X))

print(np.mean(X,axis=0))

print(np.mean(X,axis=1))

# we can also use the method version:
print(X.mean(axis=0))

# somewhat confusing, as this doesn't work:
print(X.median(axis=0))

# but this does:
print(np.median(X,axis=0))







# mini-assignment: write a function that takes as input a matrix X
# and returns a matrix that is a Z-score normalized version of X:
# that is, its columns have zero mean and standard deviation 1

def normalize(X):
    # first, demean X
    # then, make sure each column has standard deviation 1
    return X_normalized
    
# test code
X = np.random.randint(0,high=100,size=(10,3))
X_normalized = normalize(X)
print(X_normalized.mean(axis=0),X_normalized.std(axis=0))







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





# masking fancy indexing
x = np.random.randint(low=0,high=10,size=10)
flag = (x < 5)
print(flag)
print(np.where(flag))
x_subset = x[flag]
print(x,x_subset,sep='\n')




# learn more here:
# http://www.astro.washington.edu/users/vanderplas/Astr599/notebooks/11_EfficientNumpy.html