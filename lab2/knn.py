import numpy as np
from sklearn import datasets
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum( (x1 - x2)**2 ))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)


    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_closest = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.Y_train[i] for i in k_closest]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


if __name__ == '__main__':
    iris = datasets.load_iris()
    X, Y = iris.data, iris.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    clf = KNN(k=5)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)

    acc = np.sum(predictions == Y_test) / len(Y_test)
    print(acc)




