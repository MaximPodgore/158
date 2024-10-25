import numpy as np  # Changed to np for consistency
import urllib
import scipy.optimize
import random
from sklearn import linear_model
import gzip
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

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

def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

def accuracy(predictions, y):
    return sum([1 if p == y_i else 0 for p, y_i in zip(predictions, y)]) / len(y)

def BER(predictions, y):
    # Calculate true positives, false positives, true negatives, false negatives
    tp = sum([1 for p, y_i in zip(predictions, y) if p == True and y_i == True])
    tn = sum([1 for p, y_i in zip(predictions, y) if p == False and y_i == False])
    fp = sum([1 for p, y_i in zip(predictions, y) if p == True and y_i == False])
    fn = sum([1 for p, y_i in zip(predictions, y) if p == False and y_i == True])
    
    # Calculate true positive rate and true negative rate
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calculate BER
    return 1 - (tpr + tnr) / 2

# Read and parse the bankruptcy data
f = open("5year.arff", 'r')
while not '@data' in f.readline():
    pass

dataset = []
for l in f:
    if '?' in l:  # Missing entry
        continue
    l = l.split(',')
    values = [1] + [float(x) for x in l]
    values[-1] = values[-1] > 0  # Convert to bool
    dataset.append(values)

X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]

# Question 1
mod = linear_model.LogisticRegression(C=1)
mod.fit(X, y)
pred = mod.predict(X)
acc1 = accuracy(pred, y)
ber1 = BER(pred, y)
answers = {}
answers['Q1'] = convert_numpy_types([acc1, ber1])

# Question 2
mod_balanced = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod_balanced.fit(X, y)
pred_balanced = mod_balanced.predict(X)
acc2 = accuracy(pred_balanced, y)
ber2 = BER(pred_balanced, y)
answers['Q2'] = convert_numpy_types([acc2, ber2])

# Question 3
random.seed(3)
random.shuffle(dataset)

X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]

Xtrain = X[:len(X)//2]
Xvalid = X[len(X)//2:(3*len(X))//4]
Xtest = X[(3*len(X))//4:]
ytrain = y[:len(X)//2]
yvalid = y[len(X)//2:(3*len(X))//4]
ytest = y[(3*len(X))//4:]

mod_balanced = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod_balanced.fit(Xtrain, ytrain)

pred_train = mod_balanced.predict(Xtrain)
pred_valid = mod_balanced.predict(Xvalid)
pred_test = mod_balanced.predict(Xtest)

berTrain = BER(pred_train, ytrain)
berValid = BER(pred_valid, yvalid)
berTest = BER(pred_test, ytest)
answers['Q3'] = convert_numpy_types([berTrain, berValid, berTest])

# Question 4
C_values = [10**i for i in range(-4, 5)]
berList = []

for C in C_values:
    mod = linear_model.LogisticRegression(C=C, class_weight='balanced')
    mod.fit(Xtrain, ytrain)
    pred_valid = mod.predict(Xvalid)
    berList.append(BER(pred_valid, yvalid))

answers['Q4'] = convert_numpy_types(berList)

# Question 5
bestC = C_values[np.argmin(berList)]
mod_best = linear_model.LogisticRegression(C=bestC, class_weight='balanced')
mod_best.fit(Xtrain, ytrain)
pred_test = mod_best.predict(Xtest)
ber5 = BER(pred_test, ytest)
answers['Q5'] = convert_numpy_types([bestC, ber5])

# Question 6
f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(eval(l))

dataTrain = dataset[:9000]
dataTest = dataset[9000:]

# Data structures for recommendation system
usersPerItem = defaultdict(set)
itemsPerUser = defaultdict(set)
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
ratingDict = {}

for d in dataTrain:
    user = d['user_id']
    item = d['book_id']
    rating = d['rating']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    reviewsPerUser[user].append((item, rating))
    reviewsPerItem[item].append((user, rating))
    ratingDict[(user, item)] = rating

def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom if denom > 0 else 0

def mostSimilar(i, N):
    similarities = []
    users_i = usersPerItem[i]
    
    for j in usersPerItem:
        if j != i:
            similarities.append((Jaccard(users_i, usersPerItem[j]), j))
    
    return sorted(similarities, reverse=True)[:N]

answers['Q6'] = convert_numpy_types(mostSimilar('2767052', 10))

# ... (previous code remains the same until Question 7)

# Helper functions
def calculate_global_mean():
    all_ratings = [rating for _, ratings in reviewsPerItem.items() 
                  for _, rating in ratings]
    return np.mean(all_ratings) if all_ratings else 3.0

def calculate_item_mean(item):
    if item not in reviewsPerItem or not reviewsPerItem[item]:
        return calculate_global_mean()
    return np.mean([r for _, r in reviewsPerItem[item]])

def calculate_user_mean(user):
    if user not in reviewsPerUser or not reviewsPerUser[user]:
        return calculate_global_mean()
    return np.mean([r for _, r in reviewsPerUser[user]])

# Question 7 - Item-based prediction with modified implementation
def predictRating(user, item):
    # Handle base cases
    if item not in usersPerItem:
        return calculate_global_mean()
        
    item_mean = calculate_item_mean(item)
    
    if user not in itemsPerUser:
        return item_mean
        
    # Calculate weighted deviations
    weighted_diffs = []
    similarity_weights = []
    
    # Get all items rated by this user
    user_items = [i for i, _ in reviewsPerUser[user] if i != item]
    
    for other_item in user_items:
        # Calculate Jaccard similarity
        similarity = Jaccard(usersPerItem[item], usersPerItem[other_item])
        
        if similarity > 0:
            # Get the rating and mean for comparison item
            user_rating = ratingDict.get((user, other_item))
            item_rating_mean = calculate_item_mean(other_item)
            
            if user_rating is not None:
                # Store the weighted difference from mean
                weighted_diffs.append(similarity * (user_rating - item_rating_mean))
                similarity_weights.append(similarity)
    
    # If no valid comparisons found, return item mean
    if not similarity_weights:
        return item_mean
        
    # Calculate prediction using weighted average of deviations
    weighted_adjustment = sum(weighted_diffs) / sum(similarity_weights)
    prediction = item_mean + weighted_adjustment
    
    return max(1, min(5, prediction))

# Calculate MSE for item-based approach
def calculate_mse(predict_func):
    errors = []
    for d in dataTest:
        user, item = d['user_id'], d['book_id']
        actual = d['rating']
        predicted = predict_func(user, item)
        errors.append((predicted - actual) ** 2)
    return np.mean(errors) if errors else 0

mse7 = calculate_mse(predictRating)
answers['Q7'] = convert_numpy_types(mse7)

# Question 8 - User-based prediction with modified implementation
def predictRatingUser(user, item):
    # Handle base cases
    if item not in usersPerItem:
        return calculate_global_mean()
        
    if user not in itemsPerUser:
        return calculate_item_mean(item)
    
    # Find users who rated this item
    item_raters = [u for u, _ in reviewsPerItem[item] if u != user]
    
    if not item_raters:
        return calculate_user_mean(user)
    
    # Calculate weighted ratings
    weighted_ratings = []
    similarity_weights = []
    
    for other_user in item_raters:
        similarity = Jaccard(itemsPerUser[user], itemsPerUser[other_user])
        
        if similarity > 0:
            rating = ratingDict.get((other_user, item))
            if rating is not None:
                weighted_ratings.append(similarity * rating)
                similarity_weights.append(similarity)
    
    # If no valid ratings found, return user mean
    if not similarity_weights:
        return calculate_user_mean(user)
    
    # Calculate prediction using weighted average
    prediction = sum(weighted_ratings) / sum(similarity_weights)
    
    return max(1, min(5, prediction))

# Calculate MSE for user-based approach
mse8 = calculate_mse(predictRatingUser)
answers['Q8'] = convert_numpy_types(mse8)

# Write answers to file
f = open("answers_hw2.txt", 'w')
f.write(str(convert_numpy_types(answers)) + '\n')
f.close()