import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler



X = np.genfromtxt("data/x_train.txt", delimiter=None)
Y = np.genfromtxt("data/y_train.txt", delimiter=None)

X,Y = ml.shuffleData(X,Y)
Xtr, Xva, Ytr, Yva = ml.splitData(X,Y, 0.80)




# scaler = StandardScaler().fit(Xtr)
# X_train_scale = scaler.transform(Xtr)
# X_val_scale = scaler.transform(Xva)


ErrVal = []
ErrTrain = []
for reg in np.linspace(0.1, 5, 50):
    clf = LogisticRegression(C=reg)
    clf.fit(Xtr, Ytr)
    ErrVal.append(np.mean(clf.predict(Xva) != Yva))
    ErrTrain.append(np.mean(clf.predict(Xtr) != Ytr))

plt.plot(np.linspace(0.1, 5, 50), ErrTrain, 'r')
plt.plot(np.linspace(0.1, 5, 50), ErrVal, 'g')
plt.show()



# clf = LogisticRegression(C=10)
# clf.fit(X_train_scale, Ytr)
# predictions_val = clf.predict(X_val_scale)
# print 'Accuracy', np.mean(predictions_val == Yva)
