# -*- coding: utf-8 -*-
"""
Created on Wed May 30 20:12:02 2018

@author: Jan

Python 3.5
Very simle MLP classifier
"""
import numpy as np
import matplotlib.pyplot as plt
from util import get_MNIST

def softmax(x):
    exp = np.exp(x)
    try:
        return exp/exp.sum(axis=1, keepdims=True)
    except:    
        return exp/exp.sum()
    
class MultilayerPerceptron(object):
    
    def __init__(self, M, learning_rate=0.001, epochs=200, verbose=True):
        self.M = M
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        
        
    def fit(self, X, y):
        N, D = X.shape        
        self.K = len(set(y))
        self.D = D
        self.W1 = np.random.randn(self.D, self.M)
        self.b1 = np.random.randn(self.M)
        self.W2 = np.random.randn(self.M, self.K)
        self.b2 = np.random.randn(self.K)
        
        T = np.zeros((N, self.K))
        for i in range(N):
            T[i, y[i]] = 1
        
        costs = []
        
        for epoch in range(self.epochs):
            output, hidden = self.predict_proba(X, ret_Z=True)
            if epoch % 100 == 0:
                c = self.cost(T, output)
                costs.append(c)
                P = np.argmax(output, axis=1)
                r = self.classification_rate(y, P)
                
                if self.verbose:
                    print("Cost:" , c, " Accuracy: ", r)
                
            self.W2 += self.learning_rate * self.derivative_w2(hidden, T, output)
            self.b2 += self.learning_rate * self.derivative_b2(T, output)
            self.W1 += self.learning_rate * self.derivative_w1(X, hidden, T, output, self.W2)
            self.b1 += self.learning_rate * self.derivative_b1(T, output, self.W2, hidden)
            
    def cost(self, T, Y):
        tot = T*np.log(Y)
        return tot.sum()
        
    def derivative_w2(self, hidden, T, output):
        return hidden.T.dot(T-output)
            
    def derivative_b2(self, T, output):
        return (T - output).sum()
        
    def derivative_w1(self, X, hidden, T, output, W2):
        dZ = (T - output).dot(W2.T)*hidden*(1-hidden)
        return X.T.dot(dZ)
    
    def derivative_b1(self, T, output, W2, hidden):
        return ((T - output).dot(W2.T)*hidden*(1-hidden)).sum(axis=0)
        
    def classification_rate(self, Y, P):
        return np.mean(np.equal(Y, P))
        
    def predict_proba(self, X, ret_Z=False):
        Z = 1/(1+np.exp(-X.dot(self.W1)-self.b1))
        A = Z.dot(self.W2)+self.b2
        self.A = A
        if ret_Z:
            return softmax(A), Z 
        else:
            return softmax(A)
        
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
        
    def score(self, X, y):
        return np.mean(np.equal(self.predict(X), y))

if __name__ == '__main__':
    
    M = 100

#    Nclass = 500
#    X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
#    X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
#    X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
#    X = np.vstack([X1, X2, X3])
#    
#    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
#    
#    plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
#    plt.show()
#        
#    NN = MultilayerPerceptron(D, M, K)
#    NN.fit(X, Y)
#    p = NN.predict_proba(X)

    X, Y = get_MNIST(2000)
    Ntrain = int(len(Y) / 2)
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    
    NN = MultilayerPerceptron(M)
    NN.fit(Xtrain, Ytrain)
    
