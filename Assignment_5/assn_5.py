#-----------------------
# Importing dependancies
#-----------------------
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA

#-----------------------
# Reading the data
#-----------------------

data = pd.read_csv('Breast Cancer Wisconsin Data Set')

data = data.drop(columns=['id', 'Unnamed: 32'])
data = data.replace(['M','B'],[0, 1])

#-----------------------
# Scaling the data
#-----------------------

Min_max = preprocessing.MinMaxScaler(feature_range = (0,1))

cols = data.columns

# print(data)
data[cols] = Min_max.fit_transform(data[cols])

# print(data)

#-----------------------
# Splitting the data
#-----------------------

X = data.drop(columns=['diagnosis']).astype(float).to_numpy()

y = data['diagnosis'].astype(int).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

#-----------------------
# 2 Best Classifiers for Dataset
#-----------------------

# SVC Kernel=Linear

print("\nSVC Kernel Linear")

svc_lin_clf = svm.SVC(kernel='linear')

start = time.time()
svc_lin_clf.fit(X_train, y_train)
end = time.time()

print("\nTraining Time = ", end-start)
print("Training accuracy =", svc_lin_clf.score(X_train, y_train))

start = time.time()
svc_lin_clf.fit(X_test, y_test)
end = time.time()

print("\nTesting Time = ", end-start)
print("Testing accuracy =", svc_lin_clf.score(X_test, y_test))

pre_rec_fsc = precision_recall_fscore_support(y_test, svc_lin_clf.predict(X_test))

print("\nMalignant Precision =", pre_rec_fsc[0][0], ", Benign Precision =", pre_rec_fsc[0][1])
print("\nMalignant Recall =", pre_rec_fsc[1][0], ", Benign Recall =", pre_rec_fsc[1][1])
print("\nMalignant F-Score =", pre_rec_fsc[2][0], ", Benign F-Score =", pre_rec_fsc[2][1])

# Guassian Naive Bayes

print("\nGuassian Naive Bayes")

gnb_clf = GaussianNB()

start = time.time()
gnb_clf.fit(X_train, y_train)
end = time.time()

print("\nTraining Time = ", end-start)
print("Training accuracy =", gnb_clf.score(X_train, y_train))

start = time.time()
gnb_clf.fit(X_test, y_test)
end = time.time()

print("\nTesting Time = ", end-start)
print("Testing accuracy =", gnb_clf.score(X_test, y_test))

pre_rec_fsc = precision_recall_fscore_support(y_test, gnb_clf.predict(X_test))

print("\nMalignant Precision =", pre_rec_fsc[0][0], ", Benign Precision =", pre_rec_fsc[0][1])
print("\nMalignant Recall =", pre_rec_fsc[1][0], ", Benign Recall =", pre_rec_fsc[1][1])
print("\nMalignant F-Score =", pre_rec_fsc[2][0], ", Benign F-Score =", pre_rec_fsc[2][1])

#-----------------------
# Feature Selection
#-----------------------

print("\n\nFeature Selection")

X_best = SelectKBest(chi2, k=10).fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_best, y, test_size = 0.3, random_state = 42)

# SVC Kernel=Linear

print("\nSVC Kernel Linear")

svc_lin_clf = svm.SVC(kernel='linear')

start = time.time()
svc_lin_clf.fit(X_train, y_train)
end = time.time()

print("\nTraining Time = ", end-start)
print("Training accuracy =", svc_lin_clf.score(X_train, y_train))

start = time.time()
svc_lin_clf.fit(X_test, y_test)
end = time.time()

print("\nTesting Time = ", end-start)
print("Testing accuracy =", svc_lin_clf.score(X_test, y_test))

pre_rec_fsc = precision_recall_fscore_support(y_test, svc_lin_clf.predict(X_test))

print("\nMalignant Precision =", pre_rec_fsc[0][0], ", Benign Precision =", pre_rec_fsc[0][1])
print("\nMalignant Recall =", pre_rec_fsc[1][0], ", Benign Recall =", pre_rec_fsc[1][1])
print("\nMalignant F-Score =", pre_rec_fsc[2][0], ", Benign F-Score =", pre_rec_fsc[2][1])

# Guassian Naive Bayes

print("\nGuassian Naive Bayes")

gnb_clf = GaussianNB()

start = time.time()
gnb_clf.fit(X_train, y_train)
end = time.time()

print("\nTraining Time = ", end-start)
print("Training accuracy =", gnb_clf.score(X_train, y_train))

start = time.time()
gnb_clf.fit(X_test, y_test)
end = time.time()

print("\nTesting Time = ", end-start)
print("Testing accuracy =", gnb_clf.score(X_test, y_test))

pre_rec_fsc = precision_recall_fscore_support(y_test, gnb_clf.predict(X_test))

print("\nMalignant Precision =", pre_rec_fsc[0][0], ", Benign Precision =", pre_rec_fsc[0][1])
print("\nMalignant Recall =", pre_rec_fsc[1][0], ", Benign Recall =", pre_rec_fsc[1][1])
print("\nMalignant F-Score =", pre_rec_fsc[2][0], ", Benign F-Score =", pre_rec_fsc[2][1])

#-----------------------
# Principal Component Analysis
#-----------------------

print("\n\nPrincipal Component Analysis (n=30)")

pca = PCA(n_components=30)
pca.fit(X)

# exp_var_pca = np.arange(0, len(pca.explained_variance_ratio_), step = 1)
cum_sum_eigenvalues = np.cumsum(pca.explained_variance_ratio_)

print("\nSingular Values:\n", pca.singular_values_)
print("\nExplained Variance Ratio:\n", pca.explained_variance_ratio_)
print("\nCumulative Variance:\n", cum_sum_eigenvalues)

# plt.figure(1)
# plt.xticks(np.arange(0, 30, 1), fontsize=8)
# plt.plot(exp_var_pca, cum_sum_eigenvalues)
# plt.xlabel('Number of components')
# plt.ylabel('Cumulative variance')
# plt.axhline(y=0.7, color='r', label="Variance = 0.7")
# plt.legend(loc='lower right')

exp_var_pca = pca.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)

plt.figure(1)
plt.xticks(np.arange(0, 30, 1), fontsize=8)
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.axhline(y=0.7, color='r', label="Variance = 0.7")
plt.legend()

print("\n\nPrincipal Component Analysis (n=2)")

pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.fit_transform(X)
print("\nExplained Variance Ratio:\n", pca.explained_variance_ratio_)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y,  test_size = 0.3, random_state = 42)

# SVC Kernel=Linear

print("\nSVC Kernel Linear")

svc_lin_clf = svm.SVC(kernel='linear')

start = time.time()
svc_lin_clf.fit(X_train, y_train)
end = time.time()

print("\nTraining Time = ", end-start)
print("Training accuracy =", svc_lin_clf.score(X_train, y_train))

start = time.time()
svc_lin_clf.fit(X_test, y_test)
end = time.time()

print("\nTesting Time = ", end-start)
print("Testing accuracy =", svc_lin_clf.score(X_test, y_test))

pre_rec_fsc = precision_recall_fscore_support(y_test, svc_lin_clf.predict(X_test))

print("\nMalignant Precision =", pre_rec_fsc[0][0], ", Benign Precision =", pre_rec_fsc[0][1])
print("\nMalignant Recall =", pre_rec_fsc[1][0], ", Benign Recall =", pre_rec_fsc[1][1])
print("\nMalignant F-Score =", pre_rec_fsc[2][0], ", Benign F-Score =", pre_rec_fsc[2][1])

# Guassian Naive Bayes

print("\nGuassian Naive Bayes")

gnb_clf = GaussianNB()

start = time.time()
gnb_clf.fit(X_train, y_train)
end = time.time()

print("\nTraining Time = ", end-start)
print("Training accuracy =", gnb_clf.score(X_train, y_train))

start = time.time()
gnb_clf.fit(X_test, y_test)
end = time.time()

print("\nTesting Time = ", end-start)
print("Testing accuracy =", gnb_clf.score(X_test, y_test))

pre_rec_fsc = precision_recall_fscore_support(y_test, gnb_clf.predict(X_test))

print("\nMalignant Precision =", pre_rec_fsc[0][0], ", Benign Precision =", pre_rec_fsc[0][1])
print("\nMalignant Recall =", pre_rec_fsc[1][0], ", Benign Recall =", pre_rec_fsc[1][1])
print("\nMalignant F-Score =", pre_rec_fsc[2][0], ", Benign F-Score =", pre_rec_fsc[2][1])
