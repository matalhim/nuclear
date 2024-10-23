import numpy as np
import pandas as pd
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
    file_path = 'dataset/data2.txt'
    data = pd.read_csv(file_path)
    # Access the features and target variables
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Separate into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,random_state=0)

    lr = LogisticRegression(learning_rate=0.1, epochs=1000, threshold=0.5, fnum = 2)

    # Fit the model with our training data

    lr.fit(X_train, y_train)

    # Predict on our test data
    y_pred = lr.predict(X_test)

    acc = np.sum(y_pred == y_test) / len(y_test)
    print(acc)
