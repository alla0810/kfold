import random
import math
import numpy as np
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# Do not import any other libraries

def get_model(method):
    model = None
    # Implement your code to return the appropriate model with the specified parameters here
    # This is the same as Q1
    # Do NOT change the return statement
    if method == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(max_depth=10, random_state=42)
    elif method == "GaussianNB":
        model = GaussianNB()
    elif method == "LogisticRegression":
        model = LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            random_state=42,
            multi_class='multinomial'
        )
    elif method == "RandomForestClassifier":
        model = RandomForestClassifier(
            max_depth=15,
            n_estimators=250,
            random_state=42
        )
    elif method == "XGBClassifier":
        model = XGBClassifier(
            max_depth=7,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown method: {method}")
        
    return model

def my_train_test(method, X, y, pi, k):
    errors = []
    # Implement your code to construct the list of errors here
    # Do NOT change the return statement
    n = len(X)
    indices = list(range(n))

    for _ in range(k):

        idx = indices[:]
        random.shuffle(idx)

        train_size = int(pi * n)
        train_idx = idx[:train_size]
        test_idx = idx[train_size:]

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        model = get_model(method)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        wrong = np.sum(y_pred != y_test)
        error = wrong / float(len(y_test))
        errors.append(error)

    return np.array(errors)