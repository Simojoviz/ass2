from sklearn.base import BaseEstimator
import numpy as np
import statistics

class knn(BaseEstimator):

    def __init__(self,k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train.to_numpy()
        self.y_train = y_train.to_numpy()

    def __supp(self, x):

        distances = np.array([np.linalg.norm(t - x) for t in self.X_train])
        neighbors_y = np.array([ self.y_train[i] for i in np.argsort(distances)[:self.k]])

        return statistics.mode(neighbors_y)



    def predict(self, X_test):
        X_test = X_test.to_numpy()
        predict = np.array([])
        i = 0

        for x in X_test:
            i += 1
            predict = np.append(predict, self.__supp(x))
            #if (i % 100 == 0):
            #    print(f"{i} iteration")
        return predict