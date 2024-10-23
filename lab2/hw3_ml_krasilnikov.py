import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class LogisticRegression:
    def __init__(self, learning_rate=0.001, epochs=1000, threshold=0.5, fnum = 2):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.beta = np.random.normal(loc=0.0, scale=1, size=1 + fnum)
        self.threshold = threshold

    def Z(self, X):
        return self.beta[0] + np.dot(X, self.beta[1:])

    def theta(self, Z):
        return 1./(1.+np.exp(-Z))

    def predict(self, X):
        theta = self.theta(self.Z(X))
        return np.where(theta >= self.threshold, 1, 0)

    def cost(self, y, theta):
        return np.sum(np.nan_to_num(-y * np.log(theta) - (1 - y) * np.log(1 - theta)))

    def gradient_decent(self, X, y):
        for i in range(self.epochs):
            theta = self.theta(self.Z(X))
            errors = y - theta
            self.beta[1:] += self.learning_rate * X.T.dot(errors) / len(y)
            self.beta[0] += self.learning_rate * errors.sum() / len(y)
            yield self.cost(y, theta)

    def fit(self, X, y):
        self.costs = []
        for cost in self.gradient_decent(X, y):
            self.costs.append(cost)
        #return self


if __name__ == "__main__":

    # data = []

    # with open('home_3/data1.txt') as file:
    #     for line in file:
    #         x1, x2, label = line.strip().split(',')
    #         data.append([float(x1), float(x2), int(label)])

    # data = np.array(data)

    # X = data[:, :2]
    # y = data[:, 2] 

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # lr = LogisticRegression(learning_rate=0.001, epochs=1000, threshold=0.5, fnum = 2)


    # lr.fit(X_train, y_train)

    # y_pred = lr.predict(X_test)

    # acc = np.sum(y_pred == y_test) / len(y_test)
    # print(acc)

    data = []

    with open('home_3/data2.txt') as file:
        for line in file:
            x1, x2, x3, x4, label = line.strip().split(' ')
            data.append([float(x1), float(x2), float(x3), float(x4), int(label)])

    data = np.array(data)

    X = data[:, :4]
    y = data[:, 4] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    lr = LogisticRegression(learning_rate=0.001, epochs=1000, threshold=0.5, fnum = 4)


    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    acc = np.sum(y_pred == y_test) / len(y_test)
    print(acc)
