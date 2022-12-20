from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import statistics

class knn(BaseEstimator):

    def __init__(self,k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train.to_numpy()
        self.y_train = y_train.to_numpy()

    def __supp(self, x):
        distances = []  

        for i,d in enumerate([np.linalg.norm(t - x) for t in self.X_train]):
            distances.append((self.y_train[i] , d))


        neighbors_y = [ n[0] for n in sorted(list(distances),key= lambda p: p[1])[:self.k] ]

        return statistics.mode(neighbors_y)



    def predict(self, X_test):
        X_test = X_test.to_numpy()
        predict = np.array([])

        for x in X_test:
            predict = np.append(predict, self.__supp(x))

        return predict