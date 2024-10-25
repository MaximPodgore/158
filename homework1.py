import json
from matplotlib import pyplot as plt
from collections import defaultdict
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix
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

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

with open("young_adult_10000.json", "r") as f:
    dataset = [json.loads(line) for line in f]

len(dataset)
#print(len(dataset))
answers = {} # Put your answers to each question in this dictionary
dataset[0]
### Question 1
def feature(datum):
    return [datum['review_text'].count('!')]

X = np.array([feature(d) for d in dataset])
Y = np.array([d['rating'] for d in dataset])

model = LinearRegression()
model.fit(X, Y)

theta0 = model.intercept_
theta1 = model.coef_[0]
mse = mean_squared_error(Y, model.predict(X))

answers['Q1'] = [theta0, theta1, mse]
assertFloatList(answers['Q1'], 3)
### Question 2
def feature(datum):
    return [
        len(datum['review_text']),  # Length of the review
        datum['review_text'].count('!')  # Number of exclamation marks
    ]

X = np.array([feature(d) for d in dataset])
Y = np.array([d['rating'] for d in dataset])

model = linear_model.LinearRegression().fit(X,Y)

theta0 = model.intercept_
theta1, theta2 = model.coef_
mse = mean_squared_error(Y, model.predict(X))

answers['Q2'] = [theta0, theta1, theta2, mse]
assertFloatList(answers['Q2'], 4)
### Question 3
def feature(datum):
    return [datum['review_text'].count('!')]

Y = np.array([d['rating'] for d in dataset])

mses = []
for degree in range(1, 6):
    X = np.array([feature(d) for d in dataset])
    
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, Y)
    
    mse = mean_squared_error(Y, model.predict(X_poly))
    mses.append(mse)

answers['Q3'] = mses
assertFloatList(answers['Q3'], 5)
### Question 4
# Assuming dataset is sorted chronologically
train_dataset = dataset[:len(dataset)//2]
test_dataset = dataset[len(dataset)//2:]

X_train = np.array([feature(d) for d in train_dataset])
X_test = np.array([feature(d) for d in test_dataset])
Y_train = np.array([d['rating'] for d in train_dataset])
Y_test = np.array([d['rating'] for d in test_dataset])

mses = []
for degree in range(1, 6):
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, Y_train)
    
    Y_pred = model.predict(X_test_poly)
    mse = mean_squared_error(Y_test, Y_pred)
    mses.append(mse)

answers['Q4'] = mses
assertFloatList(answers['Q4'], 5)
### Question 5
Y = np.array([d['rating'] for d in dataset])

# The median minimizes the MAE for a constant predictor
best_theta0 = np.median(Y)

mae = np.mean(np.abs(Y - best_theta0))

answers['Q5'] = mae
assertFloat(answers['Q5'])

'''Beer time'''

f = open("beer_50000.json")
dataset = []
for l in f:
    if 'user/gender' in l:
        dataset.append(eval(l))

def exclamation_count(text):
    return text.count('!')

# Prepare features and labels
X = np.array([[1, exclamation_count(d['review/text'])] for d in dataset])
Y = np.array([1 if review['user/gender'] == 'Female' else 0 for review in dataset])



def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    ber = 0.5 * (fn / (fn + tp) + fp / (fp + tn))
    return [tp, tn, fp, fn, ber]

def precision_at_k(y_true, y_pred_proba, k):
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    top_k = sorted_indices[:k]
    return np.mean(y_true[top_k])

# Question 6
reg =  sklearn.linear_model.LogisticRegression(fit_intercept=False, random_state=42)
reg.fit(X, Y)
y_pred = reg.predict(X)
answers['Q6'] = calculate_metrics(Y, y_pred)
assertFloatList(answers['Q6'], 5)

# Question 7
model_balanced = LogisticRegression(class_weight='balanced',fit_intercept=False, random_state=42)
model_balanced.fit(X, Y)
y_pred_balanced = model_balanced.predict(X)
answers['Q7'] = calculate_metrics(Y, y_pred_balanced)
assertFloatList(answers['Q7'], 5)

# Question 8
y_pred_proba = model_balanced.predict_proba(X)[:, 1]
precisionList = [precision_at_k(Y, y_pred_proba, k) for k in [1, 10, 100, 1000, 10000]]
answers['Q8'] = precisionList
assertFloatList(answers['Q8'], 5)


converted_answers = convert_numpy_types(answers)
f = open("answers_hw1.txt", 'w') # Write your answers to a file
f.write(str(converted_answers) + '\n')
f.close()
 