import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import math

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

X0 = iris_X[iris_y == 0, 0:10]
X1 = iris_X[iris_y == 1, 0:10]

def euclide_distance(train_data, test_data: tuple, n):
    distance = 0
    for i in range(n):
        distance += (train_data[i] - test_data[i]) **2
    distance = math.sqrt(distance)
    return distance

class KNN:
    def __init__(self, train_data_A, train_data_B, test_data, K_test):
        self.__d = self.predict(train_data_A, train_data_B, test_data, K_test)

    def __repr__(self):
        return repr(self.__d)

    def predict(self, train_data_A, train_data_B, test_data, K_test):
        train_n = train_data_A.shape[0]
        feature_n = train_data_A.shape[1]

        distance0 = np.zeros(train_n)
        distance1 = np.zeros(train_n)
        for i in range(train_n):
            distance0[i] = euclide_distance(train_data_A[i], test_data, feature_n)
            distance1[i] = euclide_distance(train_data_B[i], test_data, feature_n)
    
        distance0 = sorted(distance0, reverse= True)
        distance1 = sorted(distance1, reverse=True)

        count0 = 0
        count1 = 0
        for i in range(K_test):
            if distance0[i] < distance1[i]:
                count0 += 1
            else:
                count1 += 1
        if count1 > count0:
            return 1
        else:
            return 0

if __name__ == "__main__":
    train_data_A = X0[0:10]
    train_data_B = X1[0:10]
    test_data = X1[0]
    print(KNN(train_data_A, train_data_B, test_data,5))


        
