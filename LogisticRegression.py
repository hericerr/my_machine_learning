# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 13:19:23 2017

@author: Jan

LogisticRegression
"""
import numpy as np
from util import get_MNIST, test_models_classification

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def cross_entropy(T, Y):
    E = 0
    for i in xrange(len(T)):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E

class LogisticRegression(object):
    
    def __str__(self):
        return "LogisticRegression"
        
    def fit(self, X, Y, learning_rate = 0.1, iters=100):
        N, D = X.shape
        X0 = np.ones((N, D+1))
        X0[:, 1:] = X 
        w = np.random.randn(D+1) / np.sqrt(D+1)       
        for t in range(iters):
            Yhat = sigmoid(X0.dot(w))
            delta = Y - Yhat
            w += learning_rate*(X0.T.dot(delta))
        self.w = w
        
    def predict(self, X):
        N, D = X.shape
        X0 = np.ones((N, D+1))
        X0[:, 1:] = X
        return sigmoid(X0.dot(self.w))
    
    def score(self, X, Y):
        P = np.round(self.predict(X))
        return np.mean(P == Y)
        
        
if __name__ == "__main__":
    X, Y = get_MNIST()
    idx = np.logical_or(Y == 0, Y == 1)
    X = X[idx]
    Y = Y[idx]

    Ntrain = len(Y) / 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = LogisticRegression()
    test_models_classification(Xtrain, Ytrain, Xtest, Ytest, [model])


