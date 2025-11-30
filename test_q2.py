# test_q2.py

from sklearn.datasets import load_digits
import numpy as np

from homework2_q2 import get_model, my_train_test


def main():

    digits = load_digits()
    X, y = digits.data, digits.target
    n = len(X)

    print(f"Number of samples: {n}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    print("-" * 60)


    pi = 0.75   
    k = 10      

    print(f"Random train-test split with pi={pi}, k={k}")
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
        errors = my_train_test(method, X, y, pi, k)
        print(f"Split errors: {errors}")
        print(f"Mean error: {errors.mean():.4f}")
        print(f"Std  error: {errors.std():.4f}")
        print("-" * 60)


if __name__ == "__main__":
    main()
