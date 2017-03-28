import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


X = np.genfromtxt("data/x_train.txt", delimiter=None)
Y = np.genfromtxt("data/y_train.txt", delimiter=None)

X,Y = ml.shuffleData(X,Y)
Xtr, Xva, Ytr, Yva = ml.splitData(X,Y, 0.80)


####finding optimal feature reduction
errTrain = []
errVal = []
ranges = [10,20,30,40,50,60,70,80,90]
for n in ranges:

    print "On feature: ", n
    pca = PCA(n_components = n)
    pca.fit(Xtr)
    tempXtr = pca.transform(Xtr)

    pcaV = PCA(n_components = n)
    pcaV.fit(Xva)
    tempXva = pca.transform(Xva)



    learner = ml.dtree.treeClassify(tempXtr, Ytr, maxDepth = 50, minParent = 2**8)
    errTrain.append(learner.err(tempXtr, Ytr))
    errVal.append(learner.err(tempXva, Yva))
plt.plot(ranges, errTrain, 'r')
plt.plot(ranges, errVal, 'g')
plt.show()
