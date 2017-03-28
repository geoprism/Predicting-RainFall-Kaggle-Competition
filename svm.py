import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from sklearn import svm

X = np.genfromtxt("data/x_train.txt", delimiter=None)
Y = np.genfromtxt("data/y_train.txt", delimiter=None)

# X,Y = ml.shuffleData(X,Y)
# Xtr, Xva, Ytr, Yva = ml.splitData(X,Y, 0.80)

Xtr = X[:10000]
Ytr = Y[:10000]
Xva = X[10000:20000]
Yva = Y[10000:20000]




estimates = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
ErrTrain = []
ErrVal = []
for i in estimates:
    print i
    neigh = svm.SVC(C = i)
    neigh.fit(Xtr, Ytr)
    ErrVal.append(np.mean(neigh.predict(Xva) != Yva))
    ErrTrain.append(np.mean(neigh.predict(Xtr) != Ytr))

plt.plot(estimates, ErrTrain, 'r')
plt.plot(estimates, ErrVal, 'g')
plt.show()


# neigh = svm.SVC()
# neigh.fit(Xtr, Ytr)
# predictions_val = neigh.predict(Xva)
# print np.mean(predictions_val == Yva)
#
#
#
# neigh = svm.SVC(C=1, gamma=0.01)
# neigh.fit(Xtr, Ytr)
# print("hi")
# predictions_val = neigh.predict(Xva)
# print np.mean(predictions_val == Yva)
