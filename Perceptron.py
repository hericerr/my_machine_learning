# -*- coding: utf-8 -*-
"""
Created on Sun May 28 17:29:33 2017

@author: Jan
"""
import numpy as np
from util import get_MNIST, test_models_classification

class Perceptron(object):
    def __str__(self):
        return "Perceptron"
    
    def fit(self, X, Y, max_epoch=1000, n=0.01):
        N, D = X.shape
        Xb = np.ones((N, D+1))
        Xb[:, 1:] = X
        self.w = np.random.randn(D+1)
        self.w[-1] = 0       
        for epoch in xrange(max_epoch):
            if epoch == max_epoch:
                break
            Yhat = self.predict(X)
            ids = Yhat != Y
            if sum(ids) == 0:
                break
            i = np.random.randint(0, sum(ids))
            x, y = Y[ids][i], Xb[ids][i, :]
            self.w += n*x*y
      
    def predict(self, X):
        N, D = X.shape
        Xb = np.ones((N, D+1))
        Xb[:, 1:] = X
        res = np.zeros(N)
        p = Xb.dot(self.w)
        i_plus = p > 0
        i_minus = p < 0
        res[i_plus] = 1
        res[i_minus] = -1
        return res

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)
        
if __name__ == '__main__':
    X, Y = get_MNIST()
    idx = np.logical_or(Y == 0, Y == 1)
    X = X[idx]
    Y = Y[idx]
    Y[Y == 0] = -1

    Ntrain = len(Y) / 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = Perceptron()
    test_models_classification(Xtrain, Ytrain, Xtest, Ytest, [model])