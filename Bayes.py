# -*- coding: utf-8 -*-
"""
Created on Wed May 17 21:04:16 2017

@author: Jan

MaximumLikelyhoodClassifier
NaiveBayes
BayesClassifier
"""
from util import get_MNIST, test_models_classification
import numpy as np
from scipy.stats import multivariate_normal as mvn

class MaximumLikelyhoodClassifier(object):
    def __str__(self):
        return "MaximumLikelyhoodClassifier"
        
    def fit(self, X, Y, smoothing=10e-3):
        self.d_gaussians = {}
        num_r, num_c = X.shape
        labels = set(Y)
        for c in labels:
            Xc = X[Y==c]
            mu , cov = Xc.mean(axis=0), np.cov(Xc.T) + np.eye(num_c)*smoothing
            self.d_gaussians[c] = {"mu":mu, "cov":cov}

    def predict(self, X):
        num_r, num_c = X.shape
        P = np.zeros((num_r, len(self.d_gaussians)))
        for c, d in self.d_gaussians.iteritems():
            mean, cov = d["mu"], d["cov"]
            P[:,c] = mvn.logpdf(X, mean=mean, cov=cov)
        return np.argmax(P, axis=1)
            
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

class NaiveBayes(object):
    def __str__(self):
        return "NaiveBayesClassifier"
        
    def fit(self, X, Y, smoothing=10e-3):
        self.d_gaussians = {}
        self.d_priors = {}
        labels = set(Y)
        for c in labels:
            Xc = X[Y==c]
            mu , var = Xc.mean(axis=0), Xc.var(axis=0) + smoothing
            self.d_gaussians[c] = {"mu":mu, "var":var}
            self.d_priors[c] = len(Y[Y==c])/(float(len(Y)))

    def predict(self, X):
        num_r, num_c = X.shape
        P = np.zeros((num_r, len(self.d_gaussians)))
        for c, d in self.d_gaussians.iteritems():
            mean, var = d["mu"], d["var"]
            P[:,c] = mvn.logpdf(X, mean=mean, cov=var) + np.log(self.d_priors[c])
        return np.argmax(P, axis=1)
        
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)
    
class BayesClassifier(object):
    def __str__(self):
        return "BayesClassifier"
        
    def fit(self, X, Y, smoothing=10e-3):
        self.d_gaussians = {}
        self.d_priors = {}
        num_r, num_c = X.shape
        labels = set(Y)
        for c in labels:
            Xc = X[Y==c]
            mu , cov = Xc.mean(axis=0), np.cov(Xc.T) + np.eye(num_c)*smoothing
            self.d_gaussians[c] = {"mu":mu, "cov":cov}
            self.d_priors[c] = len(Y[Y==c])/float(len(Y))

    def predict(self, X):
        num_r, num_c = X.shape
        P = np.zeros((num_r, len(self.d_gaussians)))
        for c, d in self.d_gaussians.iteritems():
            mean, cov = d["mu"], d["cov"]
            P[:,c] = mvn.logpdf(X, mean=mean, cov=cov) + np.log(self.d_priors[c])
        return np.argmax(P, axis=1)
            
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)
    



if __name__ == '__main__':
    X, Y = get_MNIST(10000)
    Ntrain = len(Y) / 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    models = [MaximumLikelyhoodClassifier(), NaiveBayes(), BayesClassifier()]
    test_models_classification(Xtrain, Ytrain, Xtest, Ytest, models)
