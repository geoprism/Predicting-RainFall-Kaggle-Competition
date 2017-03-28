import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


X = np.genfromtxt("data/x_train.txt", delimiter=None)
Y = np.genfromtxt("data/y_train.txt", delimiter=None)

X,Y = ml.shuffleData(X,Y)
Xtr, Xva, Ytr, Yva = ml.splitData(X,Y, 0.80)


estimates = range(1,20)
ErrTrain = []
ErrVal = []
for i in estimates:
    print i
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(Xtr, Ytr)
    ErrVal.append(np.mean(neigh.predict(Xva) != Yva))
    ErrTrain.append(np.mean(neigh.predict(Xtr) != Ytr))

plt.plot(estimates, ErrTrain, 'r')
plt.plot(estimates, ErrVal, 'g')

plt.show()


neigh = KNeighborsClassifier(n_neighbors=8)
neigh.fit(Xtr, Ytr)
predictions_val = neigh.predict(Xva)
print 'Accuracy', np.mean(predictions_val == Yva)



# clf = LogisticRegression(C=10)
# clf.fit(Xtr, Ytr)
# predictions_val = clf.predict(Xva)
# print 'Accuracy', np.mean(predictions_val == Yva)
#
#
# scaler = StandardScaler().fit(Xtr)
# X_train_scale = scaler.transform(Xtr)
# X_val_scale = scaler.transform(Xva)
# neigh = KNeighborsClassifier(n_neighbors=10)
# neigh.fit(X_train_scale, Ytr)
# predictions_val = neigh.predict(X_val_scale)
# print 'Accuracy', np.mean(predictions_val == Yva)
