# kaggle regression problem
import string
from sklearn.cross_validation import train_test_split, ShuffleSplit
from sklearn.grid_search import GridSearchCV

def stripPunct(phrase):
    phrase_no_punct = "".join(l for l in phrase if l not in string.punctuation)
    return phrase_no_punct

fn = 'data/train.tsv'
f = open(fn,'r')
sentList = []
phrases = []
sentiments = []
f.readline() # goes through the first line so as to skip it
for line in f:
    phraseId,sentId,phrase,sentiment = line.strip().split('\t')
    if sentId not in sentList:
        sentList.append(sentId)
        phrase_no_punct = stripPunct(phrase)
        phrases.append(phrase_no_punct.lower().strip())
        sentiments.append(int(sentiment))
    
from sklearn.feature_extraction.text import CountVectorizer

X = cv.fit_transform(phrases)
y = np.array(sentiments)

from sklearn.linear_model import SGDRegressor

# define the cross-validation method
numSplits = 10
cv = ShuffleSplit(len(y), n_iter=numSplits, test_size=0.5)

# now feed it all into the GridSearchCV function 
est = SGDRegressor()
parameters = {'penalty':['l2','l1','elasticnet'],'alpha':[1e-7,1e-6,1e-5],'loss':['squared_loss','huber','epsilon_insensitive']}
regr = GridSearchCV(estimator=est,param_grid=parameters,cv=cv,scoring='mean_absolute_error')

# and fit the classifier (clf)
regr.fit(X,y)

# for more details:
for param,theMean,rawScores in regr.grid_scores_:
    print(param,theMean,'\n',rawScores,'\n')
    
    
