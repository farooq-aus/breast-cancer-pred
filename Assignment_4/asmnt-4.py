#-----------------------
# Importing dependancies
#-----------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

#-----------------------
# Reading the data
#-----------------------

data = pd.read_csv('Breast Cancer Wisconsin Data Set')

X = data.drop(columns=['id', 'Unnamed: 32','diagnosis']).astype(float).to_numpy()

y = data.replace(['M','B'],[0, 1])['diagnosis'].astype(int).to_numpy()

#-----------------------
# KNN Classifier
#-----------------------

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

tr_acc = []
te_acc = []
tr_pre = []
te_pre = []
tr_rec = []
te_rec = []
tr_f1s = []
te_f1s = []
neighbors = []

print("KNN Classifier:")
for i in range(1,15):
    knn_clf = KNeighborsClassifier(n_neighbors=i)
    knn_clf.fit(X, y)
    cv_results = cross_validate(knn_clf, X, y, cv=5, scoring=scoring, return_train_score=True)
    tr_acc.append(np.average(cv_results['train_accuracy']))
    te_acc.append(np.average(cv_results['test_accuracy']))
    tr_pre.append(np.average(cv_results['train_precision_macro']))
    te_pre.append(np.average(cv_results['test_precision_macro']))
    tr_rec.append(np.average(cv_results['train_recall_macro']))
    te_rec.append(np.average(cv_results['test_recall_macro']))
    tr_f1s.append(np.average(cv_results['train_f1_macro']))
    te_f1s.append(np.average(cv_results['test_f1_macro']))
    neighbors.append(i)

plt.figure(1)
plt.plot(neighbors, tr_acc, label='Training')
plt.plot(neighbors, te_acc, label='Testing')
plt.xlabel("Neighbours")
plt.ylabel("Accuracy")
plt.legend()

plt.figure(2)
plt.plot(neighbors, tr_pre, label='Training')
plt.plot(neighbors, te_pre, label='Testing')
plt.xlabel("Neighbours")
plt.ylabel("Precision")
plt.legend()

plt.figure(3)
plt.plot(neighbors, tr_rec, label='Training')
plt.plot(neighbors, te_rec, label='Testing')
plt.xlabel("Neighbours")
plt.ylabel("Recall")
plt.legend()

plt.figure(4)
plt.plot(neighbors, tr_f1s, label='Training')
plt.plot(neighbors, te_f1s, label='Testing')
plt.xlabel("Neighbours")
plt.ylabel("F Score")
plt.legend()

knn_clf = KNeighborsClassifier(n_neighbors=13)
knn_clf.fit(X, y)
y_pred = cross_val_predict(knn_clf,X,y)
tn, fp, fn, tp = confusion_matrix(y,y_pred).ravel()
print(tn, fp, fn, tp)

cv_results = cross_validate(knn_clf, X, y, cv=5, scoring=scoring, return_train_score=True)
print(np.average(cv_results['train_accuracy']))
print(np.average(cv_results['test_accuracy']))
print(np.average(cv_results['train_precision_macro']))
print(np.average(cv_results['test_precision_macro']))
print(np.average(cv_results['train_recall_macro']))
print(np.average(cv_results['test_recall_macro']))
print(np.average(cv_results['train_f1_macro']))
print(np.average(cv_results['test_f1_macro']))

#-----------------------
# Gaussian Naive Bayes Classifier
#-----------------------

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

tr_acc = []
te_acc = []
tr_pre = []
te_pre = []
tr_rec = []
te_rec = []
tr_f1s = []
te_f1s = []

print("Gaussian Naive Bayes Classifier:")
steps = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
for step in steps:
    gnb_clf = GaussianNB(var_smoothing=step)
    gnb_clf.fit(X, y)
    cv_results = cross_validate(gnb_clf, X, y, cv=5, scoring=scoring, return_train_score=True)
    tr_acc.append(np.average(cv_results['train_accuracy']))
    te_acc.append(np.average(cv_results['test_accuracy']))
    tr_pre.append(np.average(cv_results['train_precision_macro']))
    te_pre.append(np.average(cv_results['test_precision_macro']))
    tr_rec.append(np.average(cv_results['train_recall_macro']))
    te_rec.append(np.average(cv_results['test_recall_macro']))
    tr_f1s.append(np.average(cv_results['train_f1_macro']))
    te_f1s.append(np.average(cv_results['test_f1_macro']))

plt.figure(5)
plt.plot(steps, tr_acc, label='Training')
plt.plot(steps, te_acc, label='Testing')
plt.xlabel("Var Smoothing")
plt.ylabel("Accuracy")
plt.legend()

plt.figure(6)
plt.plot(steps, tr_pre, label='Training')
plt.plot(steps, te_pre, label='Testing')
plt.xlabel("Var Smoothing")
plt.ylabel("Precision")
plt.legend()

plt.figure(7)
plt.plot(steps, tr_rec, label='Training')
plt.plot(steps, te_rec, label='Testing')
plt.xlabel("Var Smoothing")
plt.ylabel("Recall")
plt.legend()

plt.figure(8)
plt.plot(steps, tr_f1s, label='Training')
plt.plot(steps, te_f1s, label='Testing')
plt.xlabel("Var Smoothing")
plt.ylabel("F Score")
plt.legend()

gnb_clf = GaussianNB()
gnb_clf.fit(X, y)
y_pred = cross_val_predict(gnb_clf, X, y)
tn, fp, fn, tp = confusion_matrix(y,y_pred).ravel()
print(tn, fp, fn, tp)

cv_results = cross_validate(gnb_clf, X, y, cv=5, scoring=scoring, return_train_score=True)
print(np.average(cv_results['train_accuracy']))
print(np.average(cv_results['test_accuracy']))
print(np.average(cv_results['train_precision_macro']))
print(np.average(cv_results['test_precision_macro']))
print(np.average(cv_results['train_recall_macro']))
print(np.average(cv_results['test_recall_macro']))
print(np.average(cv_results['train_f1_macro']))
print(np.average(cv_results['test_f1_macro']))