# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 03:14:01 2017

@author: Jan

KMeans
SofkKMeans
GMM
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

class KMeans(object):
    
    def __str__(self):
        return "K-Means"
        
    def fit(self, X, k):
        N, D = X.shape
        idx = np.random.choice(N, size=k, replace=False)
        M = X[idx]
        while True:
            l = [KMeans.FindClosestPoint(self, point, M) for point in X]
            l = np.array(l)
            new_M = np.empty((k, D))
            for m in xrange(k):
                ids = m == l
                new_m = X[ids].mean(axis=0)
                new_M[m,:] = new_m
            
            if np.array(new_M==M).sum() == k*D:
                self.M = list(M)
                R = np.empty((N, k))
                for i in xrange(k):
                    R[:, i] = l == i
                self.R = R
                break
            else:
                M = new_M
        
    def FindClosestPoint(self, point, clusters):
        dists = []
        for cluster in clusters:
            diff = point - cluster
            d = diff.dot(diff)
            dists.append(d)
        return np.argmin(dists)
    
class SoftKMeans(object):
    
    def __str__(self):
        return "Soft K-Means"
        
    def distance(self, a, b):
        diff = a - b
        return diff.dot(diff)
    
    def fit(self, X, K, max_iter=20, beta=1.0):
        N, D = X.shape
        idx = np.random.choice(N, size=K, replace=False)
        C = X[idx]
        exponents = np.empty((N, K))
        for _ in xrange(max_iter):
            for k in xrange(K):
                for n in xrange(N):
                    exponents[n,k] = np.exp(-beta*self.distance(C[k], X[n]))
                    R = exponents / exponents.sum(axis=1, keepdims=True)

            for k in xrange(K):
                C[k] = R[:,k].dot(X) / R[:,k].sum()
                
        self.clusters = list(C)
        self.R = R
        
class GMM(object):
    
    def __str__(self):
        return "Gaussian Mixture Model"
    
    def fit(self, X, K, max_iter=20, smoothing=10e-3):
        N, D = X.shape
        idx = np.random.choice(N, size=K, replace=False)
        M = X[idx]
        C = np.zeros((K, D, D))
        pi = np.ones(K) / K
        R = np.zeros((N, K))
        for k in xrange(K):
            C[k] = np.eye(D)
        
        costs = np.zeros(max_iter)
        weighted_pdfs = np.zeros((N, K))
        
        for i in xrange(max_iter):
            for k in xrange(K):
                for n in xrange(N):
                    weighted_pdfs[n,k] = pi[k]*mvn.pdf(X[n], M[k], C[k])
    
            for k in xrange(K):
                for n in xrange(N):
                    R[n,k] = weighted_pdfs[n,k] / weighted_pdfs[n,:].sum()        
    
            for k in xrange(K):
                Nk = R[:,k].sum()
                pi[k] = Nk / N
                M[k] = R[:,k].dot(X) / Nk
                C[k] = np.sum(R[n,k]*np.outer(X[n] - M[k], X[n] - M[k]) for n in xrange(N)) / Nk + np.eye(D)*smoothing
    
    
            costs[i] = np.log(weighted_pdfs.sum(axis=1)).sum()
            if i > 0:
                if np.abs(costs[i] - costs[i-1]) < 0.1:
                    break
                
            self.pi = pi
            self.R = R
            self.M = M
            self.C = C
            
def test_clusters(X, K, models):
    for model in models:
        model.fit(X, K)
        random_colors = np.random.random((K, K))
        colors= model.R.dot(random_colors)
        plt.title(str(model))
        plt.scatter(X[:,0], X[:,1], c=colors)
        plt.show()
    
if __name__ in "__main__":
    K = 3
    
    models = [
             KMeans(),
             SoftKMeans(),
             GMM()
             ]
    
    cluster1 = np.random.multivariate_normal(mean=[5, 3], 
                cov=np.array([[1,0], [0,4]]), size=1000)
    cluster2 = np.random.multivariate_normal(mean=[0,-1], 
                cov=np.array([[1,0.8], [0.8,1.5]]), size=600)
    cluster3 = np.random.multivariate_normal(mean=[-1, 4], 
                cov=np.array([[1,0.6], [0.6,1]]), size=400)
    
    X = np.empty((2000, 2))
    X[:1000,:] = cluster1
    X[1000:1600,:] = cluster2
    X[1600:,:] = cluster3
    
    plt.scatter(X[:,0], X[:,1])
    plt.title("Raw data")
    plt.show() 
    
    plt.title("True clusters")
    plt.plot(X[:1000,0], X[:1000,1], "o")
    plt.plot(X[1000:1600,0], X[1000:1600,1], "o")
    plt.plot(X[1600:,0], X[1600:,1], "o")
    plt.show()
    
    test_clusters(X, K, models)

    D = 2
    s = 4
    mu1 = np.array([0, 0])
    mu2 = np.array([s, s])
    mu3 = np.array([0, s])

    N = 2000
    X = np.zeros((N, D))
    X[:1200, :] = np.random.randn(1200, D)*2 + mu1
    X[1200:1800, :] = np.random.randn(600, D) + mu2
    X[1800:, :] = np.random.randn(200, D)*0.5 + mu3   

    plt.scatter(X[:,0], X[:,1])
    plt.title("Raw data")
    plt.show()
    
    plt.title("True clusters")
    plt.plot(X[:1200,0], X[:1200,1], "o")
    plt.plot(X[1200:1800,0], X[1200:1800,1], "o")
    plt.plot(X[1800:,0], X[1800:,1], "o")
    plt.show()
    
    test_clusters(X, K, models)
