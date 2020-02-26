

import pandas as pd
import math as mt
import numpy as np

def sigmoid(x):
    if x > 500:
        return 1
    elif x < -500:
        return 0
    else:
        return 1/(1 + mt.exp(-x))

def calc_i(xi, yi, beta_a):
    ai = np.dot(xi, beta_a)
    pi = sigmoid(ai)
    wi = pi * (1 - pi)
    zi = ai + (yi - pi)/wi
    return np.array([wi, zi, ai])
    
def subd_k(k, y, X, beta, beta_a, n, p):
    w_rho = 0
    b = 0
    for i in range(0, n):
        r = 0
        c = calc_i(X[i,:], y[i], beta_a)
        for j in range(0, p):
            if j != k:
                r = r + X[i,j] * beta[j]
        w_rho = w_rho + c[0] * X[i,k] * (c[1] - r)
        b = b + c[0] * X[i,k]**2
        return np.array([w_rho, b])
        
def eval_subd(subd, lam):
    if subd[0] < -lam:
        return (subd[0] + lam)/subd[1]
    elif subd[0] > lam:
        return (subd[0] - lam)/subd[1]
    else:
        return 0

def coord_descent(lam, y, X, beta0, beta_a, n, p):
    beta = beta0
    for j in range(0, p):
        subd = subd_k(j, y, X, beta, beta_a, n, p)
        beta[j] = eval_subd(subd, lam)
    while np.dot(beta - beta0, beta - beta0) > 0.001:
        beta0 = beta
        for j in range(0, p):
            subd = subd_k(j, y, X, beta, beta_a, n, p)
            beta[j] = eval_subd(subd, lam)
    return beta

def iter_coord(lam, y, X, beta0, n, p):
    beta_a = beta0
    beta = coord_descent(lam, y, X, beta_a, beta_a, n, p)
    while np.dot(beta - beta_a, beta - beta_a) > 0.001:
        beta_a = beta
        beta = coord_descent(lam, y, X, beta_a, beta_a, n, p)
    return beta

X = np.array(pd.read_csv('C:/ram/projects/B.csv', header = None))
y = pd.read_csv('C:/ram/projects/labels-B.csv', header = None)
y = np.array((y.iloc[:,0]).tolist())
y = np.array(y)
n = X.shape[0]
p = X.shape[1]
beta = np.random.uniform(0, 0.5, p)
lam = 0.001

iter_coord(lam, y, X, beta, n, p)
