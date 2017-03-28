import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


X = np.genfromtxt("data/x_train.txt", delimiter=None)
Y = np.genfromtxt("data/y_train.txt", delimiter=None)





classifiers = []

tree = DecisionTreeClassifier(max_depth=25, min_samples_leaf=5)
tree.fit(X, Y)
classifiers.append(tree)

neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(X, Y)
classifiers.append(neigh)

grad = GradientBoostingClassifier(n_estimators=1500)
grad.fit(X, Y)
classifiers.append(grad)

ada_clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=25, min_samples_leaf=5), n_estimators=12)
ada_clf.fit(X, Y)
classifiers.append(ada_clf)

clf = LogisticRegression(C=10)
clf.fit(X, Y)
classifiers.append(clf)


nBag = 10
for j in range(nBag):
    print j
    Xi, Yi = ml.bootstrapData(X, Y)

    tree = DecisionTreeClassifier(max_depth=25, min_samples_leaf=5)
    tree.fit(Xi, Yi)
    classifiers.append(tree)

    neigh = KNeighborsClassifier(n_neighbors=10)  ###NEW KNN VALUE DUDE
    neigh.fit(Xi, Yi)
    classifiers.append(neigh)

    grad = GradientBoostingClassifier(n_estimators=1500)
    grad.fit(Xi, Yi)
    classifiers.append(grad)

    ada_clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=25, min_samples_leaf=5), n_estimators=12)
    ada_clf.fit(Xi, Yi)
    classifiers.append(ada_clf)

    clf = LogisticRegression(C=10)
    clf.fit(Xi, Yi)
    classifiers.append(clf)

Xte = np.genfromtxt("data/x_test.txt", delimiter=None)

Ypred = []
print "# of Classifiers: ", len(classifiers)
for j in range(len(classifiers)):
    print j
    Ypred.append(classifiers[j].predict_proba(Xte))
Ypred = np.mean(Ypred, axis=0)








































#
