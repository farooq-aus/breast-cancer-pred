"""

Machine Learning and Data Mining
CMP466_TeamProject_Assignment 2

Hamad AlMaqoodi 76307
Farooq Mirza 80205
Omar Fayed 69702


"""

from sklearn.datasets import load_breast_cancer
from sklearn import tree
from sklearn.model_selection import train_test_split

data = load_breast_cancer(return_X_y=False)


X, y = load_breast_cancer(return_X_y=True)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

#tree.plot_tree(clf)



#-----------------------
# Random splitting

print("Results of Random Splits\n\n")

for i in range(0, 15):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    print("Split #", i, end="\t")
    print("Training accuracy: ", clf.score(X_train, y_train))
    print("Testing accuracy: ", clf.score(X_test, y_test), "\n")



#-------------------------
# K-fold cross validation

from sklearn.model_selection import cross_validate

print("k-fold Cross Validation Results")

cv_results = cross_validate(clf, X, y, cv=5)
print("Cross validation results (5-fold):\n", cv_results)

cv_results = cross_validate(clf, X, y, cv=10)
print("Cross validation results (10-fold):\n", cv_results)



#-------------------------
# Decision tree classifier

""" 

Upon observing the tree we can see that the original tree
obtained by the following code has a depth of 8.

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    tree.plot_tree(clf)

We perform random splits of 80-20 and test the data with
decision trees of different depths from 1 till 8

"""

# clf = tree.DecisionTreeClassifier(max_depth=1)
# clf = tree.DecisionTreeClassifier(max_depth=2)
# clf = tree.DecisionTreeClassifier(max_depth=3)
# clf = tree.DecisionTreeClassifier(max_depth=4)
# clf = tree.DecisionTreeClassifier(max_depth=5)
# clf = tree.DecisionTreeClassifier(max_depth=6)
# clf = tree.DecisionTreeClassifier(max_depth=7)
clf = tree.DecisionTreeClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf.fit(X_train, y_train)

# tree.plot_tree(clf)

print("Training accuracy: ", clf.score(X_train, y_train))
print("Testing accuracy: ", clf.score(X_test, y_test), "\n")

"""

After noting the values in an excel sheet we plot them and
can easily observe that after the 4th depth the testing accuracy
starts to decline, this is most likely to be due to overfitting.
The training for level 4 is at 98%, and every other after it is 100%.
This means that at level 5 and after, the training accuracy shows that
the tree is overfitted.

From this we infered that at depth of 4, the tree is best-fitted.


(
    Original Tree, Level 4 and Plot of Testing/Training Accuracy vs Depth are
    attached together with this file in the .zip folder
)

"""