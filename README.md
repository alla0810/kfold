# K-Fold Cross Validation & Train–Test Split (CS412 — Digits Dataset)

This project implements **custom K-fold cross validation** and **random train–test split validation** for five classifiers using the scikit-learn **Digits dataset**.

---

## Overview

We evaluate five models:

1. DecisionTreeClassifier (max_depth=10, random_state=42)  
2. GaussianNB  
3. LogisticRegression (lbfgs, l2, multinomial, random_state=42)  
4. RandomForestClassifier (max_depth=15, n_estimators=250, random_state=42)  
5. XGBClassifier (max_depth=7, random_state=42)

These are created through a unified `get_model(method)` function.

---

## Files

| File | Description |
|------|-------------|
| `q1.py` | Implements K-fold validation: `get_model`, `get_splits`, `my_cross_val`. |
| `q2.py` | Implements random train–test splits: `get_model`, `my_train_test`. |
| `test_q1.py` | Local tester for K-fold validation. |
| `test_q2.py` | Local tester for train–test validation. |

---

## Q1 — K-Fold Cross Validation

### `get_splits(n, k, seed)`
- Randomizes indices with a fixed seed  
- Produces *k* disjoint, “almost equal” folds  
- Deterministic output for the same seed  

### `my_cross_val(method, X, y, splits)`
- Trains on k−1 folds, tests on the remaining fold  
- Computes test error  
- Returns a list of k error rates

---

## Q2 — Random Train–Test Split

### `my_train_test(method, X, y, pi, k)`
- Repeats random splitting k times  
- Uses `pi` fraction for training  
- Returns k test errors  
- Graded by mean/std comparison with reference solution

---

## Testing

Run:

```bash
python test_q1.py
python test_q2.py
```
Both scripts load the Digits dataset and report error rates for all five models.


© 2025 by **Kyo Sook Shin**  
University of Illinois Urbana-Champaign  
CS412: Introduction to Data Mining

