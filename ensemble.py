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

# X,Y = ml.shuffleData(X,Y)
# Xtr, Xva, Ytr, Yva = ml.splitData(X,Y, 0.80)


Xtr = X[:80000]
Ytr = Y[:80000]
Xva = X[80000:100000]
Yva = Y[80000:100000]




# scaler = StandardScaler().fit(X)
# X_train_scale = scaler.transform(X)
classifiers = []

tree = DecisionTreeClassifier(max_depth=25, min_samples_leaf=5)
tree.fit(Xtr, Ytr)
classifiers.append(tree)

neigh = KNeighborsClassifier(n_neighbors=10)  ###NEW KNN VALUE DUDE
neigh.fit(Xtr, Ytr)
classifiers.append(neigh)

grad = GradientBoostingClassifier(n_estimators=1300)
grad.fit(Xtr, Ytr)
classifiers.append(grad)

ada_clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=25, min_samples_leaf=5), n_estimators=12)
ada_clf.fit(Xtr, Ytr)
classifiers.append(ada_clf)

clf = LogisticRegression(C=10)
clf.fit(Xtr, Ytr)
classifiers.append(clf)

# nBag = 5
# #classifiers = []
# for j in range(nBag):
#     print j
#     Xi, Yi = ml.bootstrapData(Xtr, Ytr)
#
#     tree = DecisionTreeClassifier(max_depth=10)
#     tree.fit(Xi, Yi)
#     classifiers.append(tree)
#
#     neigh = KNeighborsClassifier(n_neighbors=8)  ###NEW KNN VALUE DUDE
#     neigh.fit(Xi, Yi)
#     classifiers.append(neigh)
#
#     grad = GradientBoostingClassifier(n_estimators=1000)
#     grad.fit(Xi, Yi)
#     classifiers.append(grad)
#
#     ada_clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=10), n_estimators=12)
#     ada_clf.fit(Xi, Yi)
#     classifiers.append(ada_clf)

    # clf = LogisticRegression(C=10)
    # clf.fit(Xi, Yi)
    # classifiers.append(clf)


#Bags(5), gradient boost, knn, decision tree = 0.72125

#Bags(5), gradient boost knn, decision tree, adaboost = 0.7234
#Bags(5), gradient boost knn, decision tree, adaboost, logisticRegression = 0.7265
#Bags(5), gradient boost, dtree, adaboost =0.7216


#increased tree depth(25) =

#increased tree depth(20) =

#increased tree depth(15) =


# Xte = np.genfromtxt("data/x_test.txt", delimiter=None)
# # X_val_scale = scaler.transform(Xte)
#
Ypred = []
print "# of Classifiers: ", len(classifiers)
for j in range(len(classifiers)):
    print j
    Ypred.append(classifiers[j].predict_proba(Xva))
Ypred = np.mean(Ypred, axis=0)

stuff = Ypred[:,1] > 0.5
print 'Accuracy Ensemble: ', np.mean( stuff == Yva)
