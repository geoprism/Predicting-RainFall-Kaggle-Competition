import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier


X = np.genfromtxt("data/x_train.txt", delimiter=None)
Y = np.genfromtxt("data/y_train.txt", delimiter=None)

X,Y = ml.shuffleData(X,Y)
Xtr, Xva, Ytr, Yva = ml.splitData(X,Y, 0.80)




estimates = range(1,7)
ErrTrain = []
ErrVal = []
for i in estimates:
    print i
    neigh = GradientBoostingClassifier(max_depth= i)
    neigh.fit(Xtr, Ytr)
    ErrVal.append(np.mean(neigh.predict(Xva) != Yva))
    ErrTrain.append(np.mean(neigh.predict(Xtr) != Ytr))

plt.plot(estimates, ErrTrain, 'r')
plt.plot(estimates, ErrVal, 'g')

plt.show()

# neigh = ml.dtree.treeClassify(Xtr, Ytr, maxDepth=20, minLeaf=4)
# print "Accuracy ml tree: ", np.mean(neigh.predict(Xva) == Yva)


neigh = GradientBoostingClassifier(n_estimators=500)
neigh.fit(Xtr, Ytr)
print "Accuracy Gradient: ", np.mean(neigh.predict(Xva) == Yva)

neigh = GradientBoostingClassifier(n_estimators=250)
neigh.fit(Xtr, Ytr)
print "Accuracy Gradient: ", np.mean(neigh.predict(Xva) == Yva)



neigh = DecisionTreeClassifier(max_depth=10)
neigh.fit(Xtr, Ytr)
predictions_val = neigh.predict(Xva)
print "D tree 10: ", np.mean(predictions_val == Yva)



neigh = KNeighborsClassifier(n_neighbors=8)
neigh.fit(Xtr, Ytr)
predictions_val = neigh.predict(Xva)
print 'Accuracy KNN', np.mean(predictions_val == Yva)
