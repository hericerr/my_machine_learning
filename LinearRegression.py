# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 04:09:39 2017

@author: Jan
"""
import numpy as np
import matplotlib.pyplot as plt
from util import get_moore

class LinearRegression(object):
    
    def __init__(self, L2=0):
        self.L2 = L2
        
    def __str__(self):
        return "LinearRegression(L2=%i)"%self.L2
    
    def fit(self, X, y):
        try:
            N, D = X.shape
            X0 = np.ones((N, D+1))
            X0[:, 1:] = X            
        except:
            N, D = len(y), 1
            X0 = np.ones((N, D+1))
            X0[:, 1] = X
        self.w = np.linalg.solve(self.L2*np.eye(D+1) + X0.T.dot(X0), X0.T.dot(y ))
        
    def predict(self, X):
        try:
            N, D = X.shape
            X0 = np.ones((N, D+1))
            X0[:, 1:] = X
        except:
            N, D = len(X), 1
            X0 = np.ones((N, D+1))
            X0[:, 1] = X
        return X0.dot(self.w)
    
    def score(self, y, p):
        """R_square"""
        e = y - p
        d = y - y.mean()
        return 1 - e.dot(e)/d.dot(d)
    
class LassoRegression(object):
    
    def __init__(self, L1 = 10.0):
        self.L1 = L1
        
    def __str__(self):
        return "LassoRegression(L1=%i)"%self.L1
    
    def fit(self, X, y, learning_rate = 0.001, iters=100):
        N, D = X.shape
        X0 = np.ones((N, D+1))
        X0[:, 1:] = X 
        w = np.random.randn(D+1) / np.sqrt(D+1)       
        for t in range(iters):
            Yhat = X0.dot(w)
            delta = Yhat - y
            w = w - learning_rate*(X0.T.dot(delta) + self.L1*np.sign(w))
        self.w = w
        
    def predict(self, X):
        N, D = X.shape
        X0 = np.ones((N, D+1))
        X0[:, 1:] = X
        return X0.dot(self.w)
    
    def score(self, y, p):
        """R_square"""
        e = y - p
        d = y - y.mean()
        return 1 - e.dot(e)/d.dot(d)
    
if __name__ == '__main__':
    #SINGLE REGRESSION (MOORE)    
    X, Y = get_moore()
    
    logY = np.log(Y)

    model = LinearRegression()
    model.fit(X, logY)
    p = model.predict(X)
    
    plt.scatter(X, logY)
    plt.plot(X, p)
    plt.title("Moore's law")
    plt.xlabel("Year")
    plt.ylabel("Transistor count (log)")
    plt.show()
    
    #POLY REGRESSION
    T = 100
    x_axis = np.linspace(0, 2*np.pi, T)
    y_axis = np.sin(x_axis)*3
    
    N = 30
    idx = np.random.choice(T, size=N, replace=False)
    Xtrain = x_axis[idx]
    Ytrain = y_axis[idx] + np.random.randn(N)
    
    XtrainPoly = np.empty((N, 3))
    XtrainPoly[:,0] = Xtrain
    XtrainPoly[:,1] = Xtrain**2
    XtrainPoly[:, 2] = Xtrain**3
              
    x_axis_poly = np.empty((len(x_axis), 3))
    x_axis_poly[:,0] = x_axis
    x_axis_poly[:,1] = x_axis**2
    x_axis_poly[:,2] = x_axis**3
    
    poly = LinearRegression()
    poly.fit(XtrainPoly, Ytrain)
    p = poly.predict(x_axis_poly)
    
    plt.title("Polynomial regression")
    plt.scatter(Xtrain, Ytrain, label="Data")
    plt.plot(x_axis, p, label="Prediction")
    plt.plot(x_axis, y_axis, label="f(x)")
    plt.legend()
    plt.show()
    
    #RIDGE vs CLASSIC
    N = 50
    X = np.linspace(0, 10, N)
    Y = 0.5*X + np.random.randn(N)
    Y[-1] += 30
    Y[-2] += 30
     
    lin = LinearRegression()
    lin.fit(X, Y)
    p1 = lin.predict(X)
    
    ridge = LinearRegression(1000.0)
    ridge.fit(X, Y)
    p2 = ridge.predict(X)
    
    plt.title("ML vs MAP(Ridge)")
    plt.scatter(X, Y)
    plt.plot(X, p1, label="maximum likelihood")
    plt.plot(X, p2, label="maximum a posteriori")
    plt.legend()
    plt.show()
    
    #LASSO vs RIDGE 
    N, D = 50, 50
    X = np.ones((N, D+1))
    data = (np.random.random((N, D)) - 0.5)*10
    X[:, 1:] = data
    true_w = np.array([0, 1, 0.5, -0.5] + [0]*(D - 3))
    Y = X.dot(true_w) + np.random.randn(N)*0.5
             
    lasso = LassoRegression()
    lasso.fit(data, Y)
    w = lasso.w
    
    ridge = LinearRegression(L2=10.0)
    ridge.fit(data, Y)
    w3 = ridge.w
    
    print "final w:", w
    plt.title("Lasso vs Ridge")
    plt.plot(true_w, label='true w')
    plt.plot(w, label='w_lasso')
    plt.plot(w3, label="w_ridge")
    plt.legend()
    plt.show()
