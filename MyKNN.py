import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

class MyKNN:
    def __init__(self, k=10):
        self.k = k

    def fit(self, x, y):
        self.X = x
        self.y = y

    def predict(self, test):
        predictions = [self.calculate(points) for points in test]
        return predictions

    def calculate(self, points):
        distance = [euclidean_distance(small_x, points) for small_x in self.X]
        k_label_group = np.argsort(distance)[:self.k]
        k_label = [self.y[i] for i in k_label_group]
        most_common = Counter(k_label).most_common(1)
        return most_common[0][0]
