#-----------------------
# Reading the data
#-----------------------
import pandas as pd
import numpy as np

data = pd.read_csv('Breast Cancer Wisconsin Data Set')
print(data)

X = data.drop(columns=['id', 'Unnamed: 32','diagnosis']).astype(float).to_numpy()
print(X)

y = data.replace(['M','B'],[0, 1])['diagnosis'].astype(int).to_numpy()
print(y)

#-----------------------
# Performing random splits
#-----------------------

from sklearn import tree
from sklearn.model_selection import train_test_split

for i in range(0, 15):
    clf = tree.DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    print("Split #", i, end="\t")
    print("Training accuracy: ", clf.score(X_train, y_train))
    print("\t\tTesting accuracy: ", clf.score(X_test, y_test), "\n")

#-----------------------
# K-Fold Cross Validation
#-----------------------

from sklearn.model_selection import cross_validate

print("k-fold Cross Validation Results")

cv_results = cross_validate(clf, X, y, cv=5)
print("Cross validation results (5-fold):\n", cv_results)

print("\nAverage 5-fold test score =", np.average(cv_results['test_score']), "\n")

cv_results = cross_validate(clf, X, y, cv=10)
print("Cross validation results (10-fold):\n", cv_results)

print("\nAverage 10-fold test score =", np.average(cv_results['test_score']), "\n")

#-----------------------
# Tweaking Hyperparameters
#-----------------------

clf = tree.DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf.fit(X_train, y_train)
tree.plot_tree(clf)

import matplotlib.pyplot as plt

test_scores = []
nodes = []
for i in range(1, 8):
    clf = tree.DecisionTreeClassifier(max_depth=i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    cv_results = cross_validate(clf, X, y, cv=5)
    test_scores.append(np.average(cv_results['test_score']) * 100)
    nodes.append(clf.tree_.node_count)
print(test_scores)
print(nodes)

plt.plot(nodes, test_scores)
plt.xlabel("Number of nodes")
plt.ylabel("Test Score")

test_scores = []
splits = []
for i in range(1, 8):
    clf = tree.DecisionTreeClassifier(min_samples_split=60 * i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    cv_results = cross_validate(clf, X, y, cv=5)
    test_scores.append(np.average(cv_results['test_score']) * 100)
    splits.append(60 * i)
print(test_scores)
print(splits)

plt.plot(splits, test_scores)
plt.xlabel("Number of splits")
plt.ylabel("Test Score")

clf = tree.DecisionTreeClassifier(max_depth=4, min_samples_split=300)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf.fit(X_train, y_train)
tree.plot_tree(clf)

