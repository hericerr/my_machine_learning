# -*- coding: utf-8 -*-
"""
Created on Wed Aug 09 06:56:32 2017
@author: Jan
Python 2.7

DecisionTreeClassifier
BaggedTreesClassifier
RandomForestClassifier
DecisionTreeRegressor
BaggedTreesRegressor
RandomForestRegressor
"""
import numpy as np 
from util import get_MNIST, get_mushroom, test_models_classification, test_models_regression

class TreeNode(object):
    """Abstract TreeNode"""
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
            elif self.k == "sqrt":
                k = np.round(np.sqrt(D))
                cols = np.random.choice(range(D), int(k))
            else:
                cols = np.random.choice(range(D), self.k)
            
            max_crit = 0
            best_col = None
            best_split = None
            for col in cols:
                crit, split = self.find_split(X, Y, col)
                if crit > max_crit:
                    max_crit = crit
                    best_col = col
                    best_split = split            

            if self.stop(max_crit):
                self.col = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = self.make_prediction(Y)
            else:
                self.col = best_col
                self.split = best_split

                if self.depth == self.max_depth:
                    self.left = None
                    self.right = None
                    self.prediction = [
                        self.make_prediction(Y[X[:,best_col] < self.split]),
                        self.make_prediction(Y[X[:,best_col] >= self.split]),
                    ]
                else:
                    left_idx = (X[:,best_col] < best_split)
                    Xleft = X[left_idx]
                    Yleft = Y[left_idx]
                    self.left = self.addTreeNode(self.depth + 1, self.max_depth, self.k)
                    self.left.fit(Xleft, Yleft)

                    right_idx = (X[:,best_col] >= best_split)
                    Xright = X[right_idx]
                    Yright = Y[right_idx]
                    self.right = self.addTreeNode(self.depth + 1, self.max_depth, self.k)
                    self.right.fit(Xright, Yright)

    def find_split(self, X, Y, col):
        x_values = X[:, col]
        sort_idx = np.argsort(x_values)
        x_values = x_values[sort_idx]
        y_values = Y[sort_idx]

        boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0]
        best_split = None
        max_crit = 0
        for b in boundaries:
            split = (x_values[b] + x_values[b+1]) / 2
            crit = self.get_crit(x_values, y_values, split)
            if crit > max_crit:
                max_crit = crit
                best_split = split
        return max_crit, best_split
    
    def stop(self, max_crit):
        raise NotImplementedError
    
    def make_prediction(self, Y):
        raise NotImplementedError
        
    def get_crit(self, x, y, split):
        raise NotImplementedError
        
    def addTreeNode(self, depth, max_depth, k):
        raise NotImplementedError
    
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
    
class TreeNodeClassifier(TreeNode):
    
    def stop(self, max_crit):
        return max_crit == 0
    
    def make_prediction(self, y):
        return np.round(y.mean())
    
    def get_crit(self, x, y, split):
        """ Information gain, binary 0, 1"""
        y0 = y[x < split]
        y1 = y[x >= split]
        N = len(y)
        y0len = len(y0)
        if y0len == 0 or y0len == N:
            return 0
        p0 = float(len(y0)) / N
        p1 = 1 - p0
        return self.entropy(y) - p0*self.entropy(y0) - p1*self.entropy(y1)
    
    def entropy(self, y):
        """assume y is binary - 0 or 1"""
        N = len(y)
        s1 = (y == 1).sum()
        if 0 == s1 or N == s1:
            return 0
        p1 = float(s1) / N
        p0 = 1 - p1
        return -p0*np.log2(p0) - p1*np.log2(p1)

    def addTreeNode(self, depth, max_depth, k):
        return TreeNodeClassifier(depth, max_depth, k)
    
class TreeNodeRegressor(TreeNode):
    
    def stop(self, max_crit):
        return max_crit < 1e-7
    
    def make_prediction(self, y):
        return y.mean()
    
    def get_crit(self, x, y, split):
        """SS reduction"""
        y0 = y[x < split]
        y1 = y[x > split]
        return self.sum_of_squares(y) - self.sum_of_squares(y0) - self.sum_of_squares(y1)

    def sum_of_squares(self, y):
        return np.sum(np.square(y - y.mean()))

    def addTreeNode(self, depth, max_depth, k):
        return TreeNodeRegressor(depth, max_depth, k)
    
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
        self.root = TreeNodeClassifier(max_depth=self.max_depth, k=self.k)
        self.root.fit(X, Y)

    def predict(self, X):
        return self.root.predict(X)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

class DecisionTreeRegressor(object):
    def __init__(self, max_depth=None, k=None):
        self.max_depth = max_depth
        self.k = k
        
    def __str__(self):
        if self.max_depth is not None:
            return "DecisionTreeRegressor(max_depth="+str(self.max_depth)+", k="+str(self.k)+")"
        else:
            return "DecisionTreeRegressor(unlimited depth, k="+str(self.k)+")"

    def fit(self, X, Y):
        self.root = TreeNodeRegressor(max_depth=self.max_depth, k=self.k)
        self.root.fit(X, Y)

    def predict(self, X):
        return self.root.predict(X)

    def score(self, X, Y):
        d1 = Y - self.predict(X)
        d2 = Y - Y.mean()
        return 1 - d1.dot(d1) / d2.dot(d2)
    
class TreeEnsemble(object):
    """Abstract class, BaggedTrees,RF"""
    
    def __init__(self, B=100):
        self.B = B
        self.trees = []
    
    def fit(self, X, y):
        N = len(X)        
        for _ in xrange(self.B):
            idx = np.random.choice(N, N)
            model = self.addTree()
            model.fit(X[idx], y[idx])
            self.trees.append(model)
            
    def predict(self, X):
        N, num_mod = len(X), len(self.trees)
        P = np.zeros((N, num_mod))
        for i in xrange(num_mod):
            P[:, i] = self.trees[i].predict(X)
        return self.make_prediction(P)
    
    def addTree(self):
        raise NotImplementedError
        
    def make_prediction(self, P):
        raise NotImplementedError
        
    def score(self, X, Y):
        raise NotImplementedError
        
    def Accuracy(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)
        
    def R_square(self, X, Y):
        d1 = Y - self.predict(X)
        d2 = Y - Y.mean()
        return 1 - d1.dot(d1) / d2.dot(d2)
    
class BaggedTreesClassifier(TreeEnsemble):
    
    def __str__(self):
        return "BaggedTreesClassifier(B=%i)"%self.B
    
    def addTree(self):
        return DecisionTreeClassifier()
    
    def make_prediction(self, P):
        return np.round(P.mean(axis=1))
    
    def score(self, X, Y):
        return self.Accuracy(X, Y)
    
class BaggedTreesRegressor(TreeEnsemble):
    
    def __str__(self):
        return "BaggedTreesRegressor(B=%i)"%self.B
    
    def addTree(self):
        return DecisionTreeRegressor()
    
    def make_prediction(self, P):
        return P.mean(axis=1)
    
    def score(self, X, Y):
        return self.R_square(X, Y)
    
class RandomForestClassifier(TreeEnsemble):
    
    def __str__(self):
        return "RandomForestClassifier(B=%i)"%self.B
    
    def addTree(self):
        return DecisionTreeClassifier(k="sqrt")
    
    def make_prediction(self, P):
        return np.round(P.mean(axis=1))
    
    def score(self, X, Y):
        return self.Accuracy(X, Y)
    
class RandomForestRegressor(TreeEnsemble):
    
    def __str__(self):
        return "RandomForestRegressor(B=%i)"%self.B
    
    def addTree(self):
        return DecisionTreeRegressor(k="sqrt")
    
    def make_prediction(self, P):
        return P.mean(axis=1)
    
    def score(self, X, Y):
        return self.R_square(X, Y)
       
if __name__ == '__main__':
#    CLASSIFICATON TEST

#    MNIST 
    X, Y = get_MNIST(1000)

    #binary classification
    idx = np.logical_or(Y == 0, Y == 1)
    X = X[idx]
    Y = Y[idx]

    Ntrain = len(Y) / 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

#   MUSHROOMS
#    X, Y = get_mushroom()
#    Ntrain = len(Y) / 2
#    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
#    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    
    models = [
              DecisionTreeClassifier(),
              BaggedTreesClassifier(),
              RandomForestClassifier()
              ]
    
    test_models_classification(Xtrain, Ytrain, Xtest, Ytest, models)
    
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
    
    models = [DecisionTreeRegressor(),
              BaggedTreesRegressor(),
              RandomForestRegressor(),
              ]
    
    test_models_regression(Xtrain, Ytrain, x_axis, y_axis, models, plot=True)

