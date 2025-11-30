# test_q1.py

from sklearn.datasets import load_digits
import numpy as np

from homework2_q1 import get_model, get_splits, my_cross_val


def main():

    digits = load_digits()
    X, y = digits.data, digits.target
    n = len(X)

    print(f"Number of samples: {n}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    print("-" * 60)


    k = 5
    seed = 42
    splits = get_splits(n, k, seed)

    print(f"K-fold splits created with k={k}, seed={seed}")
    for i, fold in enumerate(splits):
        print(f"  Fold {i}: size={len(fold)}")
    print("-" * 60)


    methods = [
        "DecisionTreeClassifier",
        "GaussianNB",
        "LogisticRegression",
        "RandomForestClassifier",
        "XGBClassifier",
    ]


    for method in methods:
        print(f"=== {method} ===")
        errors = my_cross_val(method, X, y, splits)
        print(f"Fold errors: {errors}")
        print(f"Mean error: {errors.mean():.4f}")
        print(f"Std  error: {errors.std():.4f}")
        print("-" * 60)


if __name__ == "__main__":
    main()
