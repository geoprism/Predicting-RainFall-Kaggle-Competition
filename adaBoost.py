import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


X = np.genfromtxt("data/x_train.txt", delimiter=None)
Y = np.genfromtxt("data/y_train.txt", delimiter=None)

X,Y = ml.shuffleData(X,Y)
Xtr, Xva, Ytr, Yva = ml.splitData(X,Y, 0.80)




estimates = [5,12]
# ErrTrain = []
ErrVal = []
for i in estimates:
    print i
    ada_clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=10), n_estimators=i)
    ada_clf.fit(Xtr, Ytr)
    ErrVal.append(np.mean(ada_clf.predict(Xva) == Yva))
#     ErrTrain.append(np.mean(ada_clf.predict(Xtr) != Ytr))
#
# plt.plot(estimates, ErrTrain, 'r')
plt.plot(estimates, ErrVal, 'g')
#
plt.show()


ada_clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=10), n_estimators=5)
ada_clf.fit(Xtr, Ytr)
print "Adaboost", np.mean(ada_clf.predict(Xva) == Yva)

ada_clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=10), n_estimators=12)
ada_clf.fit(Xtr, Ytr)
print "Adaboost", np.mean(ada_clf.predict(Xva) == Yva)
#
#
#
neigh = DecisionTreeClassifier(max_depth=10)
neigh.fit(Xtr, Ytr)
predictions_val = neigh.predict(Xva)
print np.mean(predictions_val == Yva)
