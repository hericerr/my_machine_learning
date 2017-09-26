import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import re
from sklearn.metrics import r2_score

def get_MNIST(limit=None):
    print "Reading in and transforming data..."
    df = pd.read_csv('train.csv')
    data = df.as_matrix()
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0 # data is from 0..255
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y

def get_xor():
    X = np.zeros((200, 2))
    X[:50] = np.random.random((50, 2)) / 2 + 0.5 # (0.5-1, 0.5-1)
    X[50:100] = np.random.random((50, 2)) / 2 # (0-0.5, 0-0.5)
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]]) # (0-0.5, 0.5-1)
    X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]]) # (0.5-1, 0-0.5)
    Y = np.array([0]*100 + [1]*100)
    return X, Y

def get_donut():
    N = 200
    R_inner = 5
    R_outer = 10

    R1 = np.random.randn(N/2) + R_inner
    theta = 2*np.pi*np.random.random(N/2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N/2) + R_outer
    theta = 2*np.pi*np.random.random(N/2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([ X_inner, X_outer ])
    Y = np.array([0]*(N/2) + [1]*(N/2))
    return X, Y

def test_models_classification(Xtrain, Ytrain, Xtest, Ytest, models):
    tt = datetime.now()
    num = 1
    for model in models:
        print ""
        print "Model%i:"%num,model
        t0 = datetime.now()
        model.fit(Xtrain, Ytrain)

        print "Training time:", (datetime.now() - t0)
        
        t0 = datetime.now()
        print "Train accuracy:", model.score(Xtrain, Ytrain)
        print "Time to compute train accuracy:", (datetime.now() - t0)
        
        t0 = datetime.now()
        print "Test accuracy:", model.score(Xtest, Ytest)
        print "Time to compute test accuracy:", (datetime.now() - t0)
        num += 1
    print ""
    print "Total duration:", (datetime.now() - tt)
    
def test_models_regression(Xtrain, Ytrain, x_axis, y_axis, models, plot=True):
    tt = datetime.now()
    num = 1
    for model in models:
        print ""
        print "Model%i"%num, model

        t0 = datetime.now()
        model.fit(Xtrain, Ytrain)
        print "Training time:", (datetime.now() - t0)
        
        t0 = datetime.now()
        prediction = model.predict(x_axis)
        print "Time to compute predictions:", (datetime.now() - t0)
        print "Accuracy:", r2_score(y_axis, prediction)
        
        if plot:
            plt.plot(x_axis, prediction)
            plt.plot(x_axis, y_axis)
            plt.show()
        num += 1
    print ""
    print "Total duration:", (datetime.now() - tt)


    
def get_moore():
    X = []
    Y = []    
    non_decimal = re.compile(r'[^\d]+')    
    for line in open('moore.txt'):
        r = line.split('\t')    
        x = int(non_decimal.sub('', r[2].split('[')[0]))
        y = int(non_decimal.sub('', r[1].split('[')[0]))
        X.append(x)
        Y.append(y)        
    return (np.array(X), np.array(Y))
    
def plot_decision_boundary(X, model):
  h = .02  # step size in the mesh
  # create a mesh to plot in
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))


  # Plot the decision boundary. For that, we will assign a color to each
  # point in the mesh [x_min, m_max]x[y_min, y_max].
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

  # Put the result into a color plot
  Z = Z.reshape(xx.shape)
  plt.contour(xx, yy, Z, cmap=plt.cm.Paired)    
    
    
    
    
    
    
