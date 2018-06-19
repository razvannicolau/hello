'''
Created on Jun 13, 2018

@author: nicolara
'''

from sklearn.datasets import fetch_mldata
from pydoc import describe
from numpy import ndarray
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
 
#from sklearn import linear_model as lin
from matplotlib import pyplot as plt
#from sklearn.cross_validation import cross_val_score
import datetime
from sklearn.preprocessing import StandardScaler

mnsit_data = 0 

def fetchData():
    print("Fetching data for MNIST");
    mnsit_data = fetch_mldata('MNIST original')
    for key in mnsit_data.keys():
        print( key )
    print( mnsit_data["DESCR"])
    print( mnsit_data["COL_NAMES"])
    print("...done fetching data");
    return mnsit_data        
    
def printImage(X, i):    
    image = X[i].reshape(28,28)
    print( image.shape );
    plt.imshow( image, alpha=1 )
    plt.show()
    
def runLogistic(X_train, X_test, y_train, y_test):   
    logReg = LogisticRegression(solver='lbfgs', penalty='l2')
    logReg.fit(X_train, y_train)
#     y_predict = logReg.predict(X_test)
    score = logReg.score(X_test, y_test)
    print( "score=" + str(score))

def runSGDClass(X_train, X_test, y_train, y_test):   
    logReg = SGDClassifier()
    logReg.fit(X_train, y_train)
#     y_predict = logReg.predict(X_test)
    score = logReg.score(X_test, y_test)
    print( "score=" + str(score))     
    
def main():
    mnsit_data = fetchData()
    X = mnsit_data.data.astype("float64")
    y = mnsit_data["target"]

# split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
#Scale the data
    stdScaler = StandardScaler()
    X_train = stdScaler.fit_transform(X_train)
    X_test = stdScaler.transform(X_test)

#     sp = skf.split(X, y)
    t1 = datetime.datetime.now()
    print("starting work: LogisticRegression")
    runLogistic(X_train, X_test, y_train, y_test)    
    t2 = datetime.datetime.now()
    duration = t2 - t1
    print("...done Duration " + str(duration.total_seconds() ))
    runSGDClass(X_train, X_test, y_train, y_test)
    t3 = datetime.datetime.now()
    duration = t3 - t2
    print("...done Duration " + str(duration.total_seconds() ))
    
main()