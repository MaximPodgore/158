import json
from matplotlib import pyplot as plt
from collections import defaultdict
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

import numpy as np
import random
import gzip
import math
import warnings

warnings.filterwarnings("ignore")
def assertFloat(x): # Checks that an answer is a float
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

with open("young_adult_10000.json", "r") as f:
    dataset = [json.loads(line) for line in f]

len(dataset)
answers = {} # Put your answers to each question in this dictionary
dataset[0]
### Question 1
def feature(datum):
    return [datum['review'].count('!')]

X = np.array([feature(d) for d in dataset])
Y = np.array([d['rating'] for d in dataset])

model = LinearRegression()
model.fit(X, Y)

theta0 = model.intercept_
theta1 = model.coef_[0]
mse = mean_squared_error(Y, model.predict(X))

answers['Q1'] = [theta0, theta1, mse]
### Question 2
def feature(datum):
    return [len(datum['review']), datum['review'].count('!')]

X = np.array([feature(d) for d in dataset])
Y = np.array([d['rating'] for d in dataset])

model = LinearRegression()
model.fit(X, Y)

theta0 = model.intercept_
theta1, theta2 = model.coef_
mse = mean_squared_error(Y, model.predict(X))

answers['Q2'] = [theta0, theta1, theta2, mse]
### Question 3
def feature(datum, deg):
    return [datum['review'].count('!')] * deg

mses = []
for degree in range(1, 6):
    X = np.array([feature(d, degree) for d in dataset])
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, Y)
    
    mse = mean_squared_error(Y, model.predict(X_poly))
    mses.append(mse)

answers['Q3'] = mses
### Question 4
# Assuming dataset is sorted chronologically
train_dataset = dataset[:len(dataset)//2]
test_dataset = dataset[len(dataset)//2:]

Y_train = np.array([d['rating'] for d in train_dataset])
Y_test = np.array([d['rating'] for d in test_dataset])

mses = []
for degree in range(1, 6):
    X_train = np.array([feature(d, degree) for d in train_dataset])
    X_test = np.array([feature(d, degree) for d in test_dataset])
    
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, Y_train)
    
    mse = mean_squared_error(Y_test, model.predict(X_test_poly))
    mses.append(mse)

answers['Q4'] = mses
### Question 5
Y = np.array([d['rating'] for d in dataset])

# The median minimizes the MAE for a constant predictor
best_theta0 = np.median(Y)

mae = np.mean(np.abs(Y - best_theta0))

answers['Q5'] = mae
### Question 6
f = open("beer_50000.json")
dataset = []
for l in f:
    if 'user/gender' in l:
        dataset.append(eval(l))
    len(dataset)
    X = 
    y = 
 
answers['Q6'] = [TP, TN, FP, FN, BER]
assertFloatList(answers['Q6'], 5)
### Question 7
 
answers["Q7"] = [TP, TN, FP, FN, BER]
assertFloatList(answers['Q7'], 5)
### Question 8
 
answers['Q8'] = precisionList
assertFloatList(answers['Q8'], 5) #List of five floats
f = open("answers_hw1.txt", 'w') # Write your answers to a file
f.write(str(answers) + '\n')
f.close()
 