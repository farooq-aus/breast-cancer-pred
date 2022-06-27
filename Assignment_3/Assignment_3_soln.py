"""

Machine Learning and Data Mining
CMP466_TeamProject_Assignment 3

Farooq Mirza 80205
Hamad AlMaqoodi 76307
Omar Fayed 69702


"""

#------------------------
#Loading Dataset
#-------------------------

import pandas as pd
import numpy as np
 
data = pd.read_csv('Breast Cancer Wisconsin Data Set')
X = data.drop(columns=['id', 'Unnamed: 32','diagnosis']).astype(float).to_numpy()
print(X)

y = data.replace(['M','B'],[0, 1])['diagnosis'].astype(int).to_numpy()
print(y)


#-------------------------
# K-fold Cross Validation
#-------------------------

from sklearn.model_selection import cross_validate

def print_cross_validate(classifier, data, labels, folds):
    cv_results = cross_validate(classifier, data, labels, cv=folds, return_train_score=True)
    print("\tCross Validation Results (", folds, "- fold ):\n", cv_results,
    "\nAverage Train Score", np.average(cv_results['train_score']),
    "\nAverage Test Score", np.average(cv_results['test_score']))


#-------------------------
# Applying Linear SVC
#-------------------------

from sklearn import svm

"""
when not specifying dual=False, the following warning is shown.
ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
"""

clf = svm.LinearSVC(dual=False)
clf.fit(X, y)

print("LinearSVC:")

cv_results = print_cross_validate(clf, X, y, 5)
cv_results = print_cross_validate(clf, X, y, 10)


#-------------------------
# SVC Linear Kernel Classifier
#-------------------------

clf = svm.SVC(kernel='linear')
clf.fit(X, y)

print("\nSVC Linear Kernel:")

cv_results = print_cross_validate(clf, X, y, 5)
cv_results = print_cross_validate(clf, X, y, 10)


#-------------------------
# SVC Polynomial Kernel Classifier
#-------------------------

def polynomial_results(arg_degree=3):
    clf = svm.SVC(kernel='poly', degree=arg_degree)
    clf.fit(X, y)

    print("\nSVC Polynomial degree", arg_degree, "Kernel:")

    cv_results = print_cross_validate(clf, X, y, 5)
    cv_results = print_cross_validate(clf, X, y, 10)

for i in range (10):
    polynomial_results(i)


#-------------------------
# SVC RBF Kernel Classifier
#-------------------------

def rbf_results(arg_C=1.0, arg_gamma='scale'):
    clf = svm.SVC(kernel='rbf', C=arg_C, gamma=arg_gamma)
    clf.fit(X, y)

    print("\nSVC RBF C =", arg_C, "gamma =", arg_gamma, "Kernel:")

    cv_results = print_cross_validate(clf, X, y, 5)
    cv_results = print_cross_validate(clf, X, y, 10)

print("\nSVC RBF default hyper-parameters:")
rbf_results()

print("\nTweaking C:\n")
for i in range(-1, 3):
    if(i==-1):
        rbf_results(0.2)
    elif(i==2):
        rbf_results(99)
    else:
        rbf_results(10 ** i)

print("\nTweaking gamma:\n")
for i in range(-4, 2):
    if(i==-4):
        rbf_results(0.0002)
    elif(i==1):
        rbf_results(9)
    else:
        rbf_results(arg_gamma = 10 ** i)

print("\nSVC RBF max value of hyper-parameters:")
rbf_results(99.99, 9.99)


#-------------------------
# SVC Sigmoid Kernel Classifier
#-------------------------

def sigmoid_results():
    clf = svm.SVC(kernel='sigmoid')
    clf.fit(X, y)

    print("\nSVC Sigmoid Kernel:")

    cv_results = print_cross_validate(clf, X, y, 5)
    cv_results = print_cross_validate(clf, X, y, 10)

sigmoid_results()