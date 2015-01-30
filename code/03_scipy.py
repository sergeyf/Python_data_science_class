# a laundry list of useful mathematical functions
import scipy.special as func

# integration
import scipy.integrate as grate

# numerical optimization
import scipy.optimize as opt



# interpolation
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

x = np.linspace(0, 10, 7)
y = np.cos(-x**2/8.0)
f1 = interp.interp1d(x, y, kind='linear')
f2 = interp.interp1d(x, y, kind='cubic')

xnew = np.linspace(0, 10, 101)

plt.plot(x,y,'o',xnew,f1(xnew),'-', xnew, f2(xnew),'--',xnew,np.cos(-xnew**2/8.0))
plt.legend(['data', 'linear', 'cubic','ground truth'], loc='best')
plt.show()


# statistics
import scipy.stats as stats

# gaussian distribution, with mean 1 and std 2
norm = stats.norm(loc=1.0,scale=2.0)

# cumulative density function at 3 points
print(norm.cdf([-10,0,10]))

# generate random samples from it
print(norm.rvs(100))




# we can also fit a distribution to data

# random samples
samples = stats.gamma.rvs(a = 1, loc=5, scale=10, size=500)
plt.hist(samples,bins=50,normed=True)
plt.show()

# fitting a gamma distribution to these samples
print(stats.gamma.fit(samples))




# statistical tests are built-in
# t-test example
def test_result(p_value):
    print("The null hypothesis is that the two samples have the same mean.")
    print("p-value is",p_value)
    if p_value < 0.05:
        print("p-value < 0.05, so we reject the null hypothesis.")
    else:
        print("p-value >= 0.05, so we fail to reject the null hypothesis.")

# same sample
samples1 = stats.norm.rvs(loc=5, scale=10, size=500)
samples2 = stats.norm.rvs(loc=5, scale=10, size=500)
t_statistics,p_value = stats.ttest_ind(samples1, samples2)
test_result(p_value)

# different samples
samples1 = stats.norm.rvs(loc=5, scale=10, size=500)
samples2 = stats.norm.rvs(loc=6.5, scale=10, size=500)
t_statistics,p_value = stats.ttest_ind(samples1, samples2)
test_result(p_value)






# other aspects of scipy we don't have time for:
'''
Fourier Transforms (scipy.fftpack)
Signal Processing (scipy.signal)
Linear Algebra (scipy.linalg)
Compressed Sparse Graph Routines (scipy.sparse.csgraph)
Spatial data structures and algorithms (scipy.spatial)
Multidimensional image processing (scipy.ndimage)
File IO (scipy.io)
Weave (scipy.weave)
'''