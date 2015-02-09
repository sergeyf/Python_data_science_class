import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster") # makes thicker lines
plt.rcParams['figure.figsize'] = (9.0, 6.0)

# scikit-learn is my favorite!
# and it comes with its own example datasets
from sklearn import datasets, svm

# iris, the classic classification dataset
iris = datasets.load_iris()

# the actual data is in "iris.data":
X = iris.data # tradition
nSamples, nFeatures = iris.data.shape
print(nSamples,nFeatures)
print(iris.feature_names)

# the labels or "targets"
y = iris.target
print(y)
print(iris.target_names)

# let's plot two of the dimensions to get a sense of the data
def classificationScatter(y,x1,x2,extras=False):
    for target in set(y):
        cond = (y == target)
        plt.plot(x1[cond],x2[cond],linestyle='none',alpha=0.7, marker='o', label=iris.target_names[target])
        
    plt.xlim([np.min(x1)-0.5,np.max(x1)+0.5])
    plt.ylim([np.min(x2)-0.5,np.max(x2)+0.5])
    if extras == True:
        plt.xlabel(iris.feature_names[0])
        plt.ylabel(iris.feature_names[1])
        plt.legend(iris.target_names,markerscale=1.25,numpoints=3)

Xsub = X[:,[1,2]] # subset
x1 = Xsub[:,0]
x2 = Xsub[:,1]
classificationScatter(y,x1,x2,extras=True)
plt.show()

# aside: why not plt.scatter?
plt.scatter(x1,x2,c=y,s=50,cmap=plt.cm.jet)
# hard to label...





# and now, immediate and total machine learning
# we create an instance of SVM and fit out data. 
# We do not scale our data since we want to plot the support vectors
C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(Xsub, y)
rbf_svc = svm.SVC(kernel='rbf', C=C).fit(Xsub, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(Xsub, y)
lin_svc = svm.LinearSVC(C=C).fit(Xsub, y)
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']
          

# now we have to plot the classification decision
# for every possible coordinate:
h = 0.01 # density of plot
x_min, x_max = x1.min() - 1, x1.max() + 1
y_min, y_max = x2.min() - 1, x2.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
                     
for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. 
    # For that, we will assign a color to each
    # point in the mesh [x_min, m_max] * [y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)

    # Plot also the training points
    classificationScatter(y,x1,x2)
    plt.title(titles[i])

plt.show()
                     
            
            
            
            
            
# let's take a few steps back!
# and get a harder dataset: handwritten
digits = datasets.load_digits()

# let's look at some
plt.imshow(digits.images[0])
plt.show()

plt.imshow(digits.images[100])
plt.show()

# but we flaten the pixels into vectors
X = digits.data
y = digits.target
nSamples, nFeatures = X.shape

# first, a classification example with k nearest neighbors
from sklearn.neighbors import KNeighborsClassifier

# instantiate a classifier with some parameters
kNN = KNeighborsClassifier(n_neighbors=7,algorithm='brute')

# fit it to the data (aka training)
kNN.fit(X,y)

# now we can get predictions!
print( kNN.predict(X[2]) )

# and see how they stack against the true labels
print( np.mean( kNN.predict(X) == y ) )

# but we can't evaluate our fit
# on the same data we trained on
# so let's split up (X,y) into a training and test set
perm = np.random.permutation(nSamples)
print(perm)

# so the first half of these permuted indices are training
nHalf = nSamples//2 # integer division
permTrain = perm[:nHalf] # first half
permTest = perm[nHalf:] # second half
print( sorted(permTrain) )
print( sorted(permTest) )

# let's split up X and y now
Xtrain = X[permTrain,:]
ytrain = y[permTrain]
Xtest = X[permTest,:]
ytest = y[permTest]

# ok great, let us retrain
kNN = KNeighborsClassifier(n_neighbors=7,algorithm='brute')
kNN.fit(Xtrain,ytrain)

# and test
print( np.mean(kNN.predict(Xtest) == ytest) )




# pretty good, but how does it depend
# on n_neighbors?
# let's check 
perm = np.random.permutation(nSamples)
print(perm)

# so the first half of these permuted indices are training
nHalf = nSamples//2 # integer division
permTrain = perm[:nHalf] # first half
permTest = perm[nHalf:] # second half


# let's split up X and y now
Xtrain = X[permTrain,:]
ytrain = y[permTrain]
Xtest = X[permTest,:]
ytest = y[permTest]

kVec = [1,3,5,7,9,11,13,15]
acc = np.zeros(len(kVec))
for i,k in enumerate(kVec):
    kNN = KNeighborsClassifier(n_neighbors=k,algorithm='brute')
    kNN.fit(Xtrain,ytrain)
    acc[i] = np.mean(kNN.predict(Xtest) == ytest) 

# and then plot the results
plt.plot(kVec,acc)
plt.xlabel('k in kNN')
plt.ylabel('Test Accuracy')
plt.ylim([np.min(acc)-0.05, 1.05])
plt.show()




# interesting!
# but could this be an artifact of our random split?
# maybe other random splits are different?
# only one way to find out...
numSplits = 10
kVec = [1,3,5,7,9,11,13,15]
acc = np.zeros([len(kVec),numSplits])

for n in range(numSplits):
    # we randomly split 10 times
    perm = np.random.permutation(nSamples)
    permTrain = perm[:nHalf] # first half
    permTest = perm[nHalf:] # second half

    Xtrain = X[permTrain,:]
    ytrain = y[permTrain]
    Xtest = X[permTest,:]
    ytest = y[permTest]
    
    # now we train the classifier for each k
    for i,k in enumerate(kVec):
        kNN = KNeighborsClassifier(n_neighbors=k,algorithm='brute')
        kNN.fit(Xtrain,ytrain)
        acc[i,n] = np.mean(kNN.predict(Xtest) == ytest) 
        
# now we can plot error bars!
plt.errorbar(kVec,np.mean(acc,axis=1),yerr=np.std(acc,axis=1))
plt.xlabel('k in kNN')
plt.ylabel('Test Accuracy')
plt.ylim([np.min(np.mean(acc,axis=1))-0.05, 1.025])
plt.show()





# that was super great
# and it turns out scikit-learn can do it all for us =)

# if all you want to do is split data
from sklearn.cross_validation import train_test_split, ShuffleSplit
Xtrain, Xtest, ytrain, y_est = train_test_split(X,y,test_size=0.5)

# and here is how to do EVERYTHING in one step
from sklearn.grid_search import GridSearchCV
# define the parameters you want to cycle over
parameters = {'n_neighbors':kVec}

# define the algorithms
kNN = KNeighborsClassifier(algorithm='brute')

# define the cross-validation method
# kFold is the most common, but i prefer shuffle and split
cv = ShuffleSplit(nSamples, n_iter=numSplits, test_size=0.5)

# now feed it all into the GridSearchCV function
clf = GridSearchCV(estimator=kNN,param_grid=parameters,cv=cv)

# and fit the classifier (clf)
clf.fit(X,y)


# that's it! if you just want the best param:
print( clf.best_params_ )

# for more details:
for param,theMean,rawScores in clf.grid_scores_:
    print(param,theMean,'\n',rawScores,'\n')
    
    
    
    

# regression!
import pandas as pd

# import the fremont bridge data again
hourly = pd.read_csv("data/fremont_bridge_data.csv", index_col='Date', parse_dates=True)
weekly = hourly.resample('w','sum')
y = np.array(weekly['Fremont Bridge NB'])

# now we have outputs but no inputs
# so we make up an X
X = np.arange(len(y))
X = X[:,np.newaxis] # has to be an array of arrays!

# quick plot of the data
plt.scatter(X,y)
plt.show()

# now we fit a linear model
from sklearn.linear_model import Ridge
regr = Ridge(alpha=0.5)
regr.fit(X, y)

# now we plot the fit
plt.scatter(X, y,label='data')
plt.plot(X, regr.predict(X),label='linear fit')
plt.legend()
plt.show()

# boring! time to do better...






# Mini-Assignment
# Do a Random Forest Regression fit instead
# google for: RandomForestRegressor 
# mess around with the following two parameters:
# n_estimators (look at 1,5,10,25,100)
# max_depth (look at 1,2,3,4,5,6,7)
# what's the best combo of n_estimators and max_depth?
# why?

# import it here
import pandas.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(n_estimators=100,max_depth=4)
regr.fit(X, y)
plt.scatter(X, y,label='data')
plt.plot(X, regr.predict(X),label='Random Forest fit')
plt.legend()
plt.show()








# and, just like with classification, we can do
# multivariate non-linear regression
boston = datasets.load_boston()
X = boston.data
y = boston.target
nSamples,nFeatures = X.shape
print( nSamples,nFeatures )

# we can cross-validate both n_estimators and max_depth
parameters = {'n_estimators':[1,5,10,25,100],'max_depth':(1,2,3,4,5,6,7,None)}

# define the algorithms
rfr = RandomForestRegressor()

# define the cross-validation method
numSplits = 10
cv = ShuffleSplit(nSamples, n_iter=numSplits, test_size=0.5)

# now feed it all into the GridSearchCV function
regr = GridSearchCV(estimator=rfr,param_grid=parameters,cv=cv,scoring='mean_absolute_error')

# and fit the classifier (clf)
regr.fit(X,y)

# and the winner is
print( regr.best_params_ )

# let's take a closer look
for i in regr.grid_scores_:
    print(i)
    
# NEGATIVE?
# we have to understand the internal workings of sklearn...
    
    



# clustering, the dark art of

# first, let's make some artificial data
from sklearn.datasets.samples_generator import make_blobs
nSamples = 750
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=nSamples, centers=centers, cluster_std=0.4)

# plot it
plt.scatter(X[:,0],X[:,1],s=50,cmap=plt.cm.Paired,alpha=0.8)

# now we try kmeans
from sklearn import cluster
km = cluster.KMeans(n_clusters=3)
km.fit(X)
plt.scatter(X[:,0],X[:,1],s=50,alpha=0.8,c=km.labels_,cmap=plt.cm.Paired)



# too easy?
X,labels = datasets.make_circles(n_samples=nSamples, factor=.5,
                                      noise=.05)

# concentric circles
plt.scatter(X[:,0],X[:,1],s=50,cmap=plt.cm.Paired,alpha=0.8)

# let's try kmeans
km = cluster.KMeans(n_clusters=2)
km.fit(X)
plt.scatter(X[:,0],X[:,1],s=50,alpha=0.8,c=km.labels_,cmap=plt.cm.Paired)

# absolutely terrible!
# let's use something fancier

# Mini-Assignment
# use SpectralClustering to deal with X
# make sure to try different settings of "affinity"
from sklearn.cluster import SpectralClustering
sc = SpectralClustering(n_clusters=2)
sc.fit(X)
plt.scatter(X[:,0],X[:,1],s=50,alpha=0.8,c=sc.labels_,cmap=plt.cm.Paired)



# and so on. Scikit-Learn is massive and great.