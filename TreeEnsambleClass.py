# -*- coding: utf-8 -*-
"""
Created on Sun May 21 12:06:47 2017

@author: Jan
"""
import numpy as np
from util import get_MNIST, get_mushroom, test_models_classification

def entropy(y):
    # assume y is binary - 0 or 1
    N = len(y)
    s1 = (y == 1).sum()
    if 0 == s1 or N == s1:
        return 0
    p1 = float(s1) / N
    p0 = 1 - p1
    return -p0*np.log2(p0) - p1*np.log2(p1)

class TreeNode(object):
    def __init__(self, depth=0, max_depth=None, k=None):
        self.depth = depth
        self.max_depth = max_depth
        self.k = k
        
    def fit(self, X, Y):
        if len(Y) == 1 or len(set(Y)) == 1:
            self.col = None
            self.split = None
            self.left = None
            self.right = None
            self.prediction = Y[0]
            
        else:
            D = X.shape[1]
            if self.k is None:
                cols = range(D)
            else:
                cols = np.random.choice(range(D), self.k)
            
            max_ig = 0
            best_col = None
            best_split = None
            for col in cols:
                ig, split = self.find_split(X, Y, col)
                if ig > max_ig:
                    max_ig = ig
                    best_col = col
                    best_split = split            

            if max_ig == 0:
                self.col = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = np.round(Y.mean())
            else:
                self.col = best_col
                self.split = best_split

                if self.depth == self.max_depth:
                    self.left = None
                    self.right = None
                    self.prediction = [
                        np.round(Y[X[:,best_col] < self.split].mean()),
                        np.round(Y[X[:,best_col] >= self.split].mean()),
                    ]
                else:
                    left_idx = (X[:,best_col] < best_split)
                    Xleft = X[left_idx]
                    Yleft = Y[left_idx]
                    self.left = TreeNode(self.depth + 1, self.max_depth, self.k)
                    self.left.fit(Xleft, Yleft)

                    right_idx = (X[:,best_col] >= best_split)
                    Xright = X[right_idx]
                    Yright = Y[right_idx]
                    self.right = TreeNode(self.depth + 1, self.max_depth, self.k)
                    self.right.fit(Xright, Yright)

    def find_split(self, X, Y, col):
        x_values = X[:, col]
        sort_idx = np.argsort(x_values)
        x_values = x_values[sort_idx]
        y_values = Y[sort_idx]

        boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0]
        best_split = None
        max_ig = 0
        for b in boundaries:
            split = (x_values[b] + x_values[b+1]) / 2
            ig = self.information_gain(x_values, y_values, split)
            if ig > max_ig:
                max_ig = ig
                best_split = split
        return max_ig, best_split

    def information_gain(self, x, y, split):
        #classes are 0 and 1
        y0 = y[x < split]
        y1 = y[x >= split]
        N = len(y)
        y0len = len(y0)
        if y0len == 0 or y0len == N:
            return 0
        p0 = float(len(y0)) / N
        p1 = 1 - p0
        return entropy(y) - p0*entropy(y0) - p1*entropy(y1)
    
    def predict_one(self, x):
        if self.col is not None and self.split is not None:
            feature = x[self.col]
            if feature < self.split:
                if self.left:
                    p = self.left.predict_one(x)
                else:
                    p = self.prediction[0]
            else:
                if self.right:
                    p = self.right.predict_one(x)
                else:
                    p = self.prediction[1]
        else:
            p = self.prediction
        return p

    def predict(self, X):
        N = len(X)
        P = np.zeros(N)
        for i in xrange(N):
            P[i] = self.predict_one(X[i])
        return P


class DecisionTreeClassifier(object):
    def __init__(self, max_depth=None, k=None):
        self.max_depth = max_depth
        self.k = k
        
    def __str__(self):
        if self.max_depth is not None:
            return "DecisionTreeClassifier(max_depth="+str(self.max_depth)+", k="+str(self.k)+")"
        else:
            return "DecisionTreeClassifier(unlimited depth, k="+str(self.k)+")"

    def fit(self, X, Y):
        self.root = TreeNode(max_depth=self.max_depth, k=self.k)
        self.root.fit(X, Y)

    def predict(self, X):
        return self.root.predict(X)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)
    
class BaggedTreesClassifier(object):
    def __init__(self, B=100):
        self.B = B
        self.trees = []
        
    def __str__(self):
        return "BaggedTreesClassifier(B=%i)"%self.B
        
    def fit(self, X, y):
        N = len(X)        
        for _ in xrange(self.B):
            idx = np.random.choice(N, N)
            model = DecisionTreeClassifier()
            model.fit(X[idx], y[idx])
            self.trees.append(model)

    def predict(self, X):
        N, num_mod = len(X), len(self.trees)
        P = np.zeros((N, num_mod))
        for i in xrange(num_mod):
            P[:, i] = self.trees[i].predict(X)
        return np.round(P.mean(axis=1))
    
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)
    
class RandomForestClassifier(object):
    def __init__(self, max_depth=None, k=15, B=100):
        self.trees = []
        self.max_depth = max_depth
        self.k = k
        self.B = B
        
    def __str__(self):
        if self.max_depth is not None:
            return "RandomForestClassifier(max_depth=%i, k=%i, B=%i)" %(self.max_depth, self.k, self.B)
        else:
            return "RandomForestClassifier(unlimited depth, k=%i, B=%i)" %(self.k, self.B)
        
    def fit(self, X, Y):
        N = len(X)        
        for _ in xrange(self.B):
            idx = np.random.choice(N, N)
            model = DecisionTreeClassifier(max_depth = self.max_depth, k =  self.k)
            model.fit(X[idx], Y[idx])
            self.trees.append(model)
            
    def predict(self, X):
        N, num_mod = len(X), len(self.trees)
        P = np.zeros((N, num_mod))
        for i in xrange(num_mod):
            P[:, i] = self.trees[i].predict(X)
        return np.round(P.mean(axis=1))
    
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)

if __name__ == '__main__':
    X, Y = get_MNIST()

    #binary classification
    idx = np.logical_or(Y == 0, Y == 1)
    X = X[idx]
    Y = Y[idx]

    Ntrain = len(Y) / 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    
#    models = [DecisionTreeClassifier(),
#              BaggedTreesClassifier(),
#              RandomForestClassifier()]
#    
#    X, Y = get_mushroom()
#    Ntrain = len(Y) / 2
#    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
#    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
#    
#    test_models_classification(Xtrain, Ytrain, Xtest, Ytest, models)