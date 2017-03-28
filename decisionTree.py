import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


X = np.genfromtxt("data/x_train.txt", delimiter=None)
Y = np.genfromtxt("data/y_train.txt", delimiter=None)

# X,Y = ml.shuffleData(X,Y)
# Xtr, Xva, Ytr, Yva = ml.splitData(X,Y, 0.80)

Xtr = X[:10000]
Ytr = Y[:10000]
Xva = X[10000:20000]
Yva = Y[10000:20000]



estimates = range(1,20)
ErrTrain = []
ErrVal = []
for i in estimates:
    print i
    neigh = DecisionTreeClassifier(max_depth= i)
    neigh.fit(Xtr, Ytr)
    ErrVal.append(np.mean(neigh.predict(Xva) != Yva))
    ErrTrain.append(np.mean(neigh.predict(Xtr) != Ytr))

plt.plot(estimates, ErrTrain, 'r')
plt.plot(estimates, ErrVal, 'g')

plt.show()



neigh = DecisionTreeClassifier(max_depth=20, min_samples_leaf=4)
neigh.fit(Xtr, Ytr)
predictions_val = neigh.predict(Xva)
print "D tree 10: ", np.mean(predictions_val == Yva)



neigh = DecisionTreeClassifier(max_depth=10)
neigh.fit(Xtr, Ytr)
predictions_val = neigh.predict(Xva)
print "D tree 10: ", np.mean(predictions_val == Yva)
