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

def get_splits(n, k, seed):
    splits = None
    # Implement your code to construct the splits here
    # Do NOT change the return statement
    if k <= 0:
        raise ValueError("k must be positive")
    if n < 0:
        raise ValueError("n must be non-negative")

    # deterministic randomization using seed
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    base_size = n // k           
    remainder = n % k            

    splits = []
    start = 0
    for i in range(k):
        fold_size = base_size + (1 if i < remainder else 0)
        end = start + fold_size
        splits.append(indices[start:end])
        start = end

    return splits

def my_cross_val(method, X, y, splits):
    errors = []
    # Implement your code to construct the list of errors here
    # Do NOT change the return statement
    n = len(X)
    all_indices = list(range(n))

    for test_idx in splits:
        test_set = set(test_idx)

        train_idx = [i for i in all_indices if i not in test_set]

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