# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 04:42:58 2017

@author: Jan
"""
import numpy as np
from sortedcontainers import SortedList
from util import get_MNIST, test_models_classification, test_models_regression

class KNN(object):
    """Abstract KNN"""
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        
    def predict(self, X, k=None):
        if k is None:
            k = self.k           
        y = np.zeros(len(X))
        for i, x in enumerate(X):
            sl = SortedList(load=k)
            for j, xt in enumerate(self.X):
                diff = x - xt
                d = diff.dot(diff)
                if len(sl) < k:
                    sl.add((d, self.y[j]))
                else:
                    if sl[-1][0] > d:
                        del sl[-1]
                        sl.add((d, self.y[j]))
            y[i] = self.decide(sl)
        return y
    
    def decide(self):
        raise NotImplementedError
        
class KNN_Classifier(KNN):
           
    def __str__(self):
        return "KNN_Classifier(k=%i)" %self.k
    
    def decide(self, sl):
        votes = {}
        for _, v in sl:
            votes[v] = votes.get(v, 0) + 1
        
        max_count = 0
        max_count_val = -1
        for v, c in votes.iteritems():
            if c > max_count:
                max_count = c
                max_count_val = v
                
        return max_count_val
    
    def score(self, X, Y, k=None):
        P = self.predict(X, k)
        return np.mean(P == Y)
    
class KNN_Regressor(KNN):
            
    def __str__(self):
        return "KNN_Regressor(k=%i)" %self.k
    
    def decide(self, sl):
        _, y = zip(*sl)
        return np.mean(y)
    
    def score(self, X, Y):
        d1 = Y - self.predict(X)
        d2 = Y - Y.mean()
        return 1 - d1.dot(d1) / d2.dot(d2)
    
if __name__ == '__main__':
#    #CLASSIFICATION TEST (MNIST)
#    X, Y = get_MNIST(2000)
#    Ntrain = len(Y) / 2
#    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
#    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
#    
#    models = [KNN_Classifier(k) for k in xrange(1,6)]
#    test_models_classification(Xtrain, Ytrain, Xtest, Ytest, models)
    
    #REGRESSION TEST
    # create the data
    T = 100
    x_axis = np.linspace(0, 2*np.pi, T)
    y_axis = np.sin(x_axis)*3
    
    # get the training data
    N = 30
    idx = np.random.choice(T, size=N, replace=False)
    Xtrain = x_axis[idx].reshape(N, 1) 
    Ytrain = y_axis[idx] + np.random.randn(N)
    
    models = [KNN_Regressor(k) for k in xrange(1,6)]
    
    test_models_regression(Xtrain, Ytrain, x_axis, y_axis, models, plot=True)
