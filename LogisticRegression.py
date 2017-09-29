# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 13:19:23 2017

@author: Jan

LogisticRegression
LogisticLasso
"""
import numpy as np
from util import get_MNIST, test_models_classification
import warnings
warnings.filterwarnings("ignore")

def sigmoid(z):
    return 1/(1 + np.exp(-z))

class LogisticRegression(object):
    
    def __init__(self, L2=0):
        self.L2 = L2
    
    def __str__(self):
        return "LogisticRegression(L2=%f)"%self.L2
        
    def fit(self, X, Y, learning_rate = 0.1, iters=100):
        N, D = X.shape
        X0 = np.ones((N, D+1))
        X0[:, 1:] = X 
        w = np.random.randn(D+1) / np.sqrt(D+1)       
        for t in range(iters):
            Yhat = sigmoid(X0.dot(w))
            delta = Y - Yhat
            w += learning_rate*(X0.T.dot(delta) - self.L2*w)
        self.w = w
        
    def predict_proba(self, X):
        N, D = X.shape
        X0 = np.ones((N, D+1))
        X0[:, 1:] = X
        return sigmoid(X0.dot(self.w))
    
    def predict(self, X):
        return np.round(self.predict_proba(X))
    
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)
    
class LogisticLasso(LogisticRegression):
    
    def __init__(self, L1=0):
        self.L1 = L1
    
    def __str__(self):
        return "LogisticRegression(L1=%f)"%self.L1
        
    def fit(self, X, Y, learning_rate = 0.1, iters=100):
        N, D = X.shape
        X0 = np.ones((N, D+1))
        X0[:, 1:] = X 
        w = np.random.randn(D+1) / np.sqrt(D+1)       
        for t in range(iters):
            Yhat = sigmoid(X0.dot(w))
            delta = Y - Yhat
            w += learning_rate*(X0.T.dot(delta) + self.L1*np.sign(*w))
        self.w = w    
        
        
if __name__ == "__main__":
    X, Y = get_MNIST()
    idx = np.logical_or(Y == 0, Y == 1)
    X = X[idx]
    Y = Y[idx]

    Ntrain = len(Y) / 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = LogisticRegression(L2=0.1)
    test_models_classification(Xtrain, Ytrain, Xtest, Ytest, [model])
