import matplotlib.pyplot as plt
import numpy as np

# let's plot!
x = np.arange(0, 4*np.pi, 0.1) # regularly spaced x-axis
y = np.sin(x) # f(x) = sin(x)
plt.plot(x, y)
plt.show()

# how ugly and small! 
# let's use the package "seaborn" to get nice plots
import seaborn as sns
sns.set_context("poster") # makes thicker lines

# and make them larger
plt.rcParams['figure.figsize'] = (10.0, 5.0)




# now replot, and add axis labels
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, y)
plt.show()




# how about more than 1 plot on the same plot?
for i in range(3):
    plt.plot(x,np.sin(x*i),label=i)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()




# plotting some points in red
plt.plot([1,2,3,4], [1,4,9,16], 'rH')
plt.axis([0, 5, 0, 20])
plt.show()



# another example
x = np.arange(0, np.pi, 0.1)
plt.plot(x, x, 'r--', x, x**2, 'bs', x, x**3, 'g^')
plt.show()





# multiple figures in one plot!
t = np.arange(0.0, 5.0, 0.02)
f = np.exp(-t) * np.cos(2*np.pi*t)

plt.subplot(211)
plt.plot(t, f, 'k')
plt.title('First plot')

plt.subplot(212)
plt.plot(t, np.cos(2*np.pi*t), 'r--')
plt.title('Second plot')
plt.show()





# a histogram example with bells and whistles
mu, sigma = 100, 15 # or so I'm told
# here is simulated IQ data sampled from a Normal distribution
x = mu + sigma * np.random.randn(10000)

# the histogram of the data and the plot, all in one
n, bins, patches = plt.hist(x, 50, normed=1)
plt.xlabel('IQ Score')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100$, $\sigma=15$',fontsize=20)
plt.show()




# seaborn builds on top of matplotlib
# and has a lot of useful functionality
# for the practicing data scientist
tips = sns.load_dataset("tips")
sns.jointplot("total_bill", "tip", tips)