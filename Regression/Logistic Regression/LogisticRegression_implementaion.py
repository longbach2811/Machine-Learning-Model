import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, X, y, weights, biases, learning_rate, iteration):
        self.X = X
        self.y = y
        self.weights = weights
        self.biases = biases
        self.learning_rate = learning_rate
        self.iteration = iteration

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def predict(self, var):
        return self.sigmoid(np.dot(self.weights, var) + self.biases)
    
    def cost_function(self):
        J = 0
        for i in range(len(self.y)):
            J += -(self.y[i] * np.log(self.predict(self.X[i])) + 
            (1 - self.y[i]) * np.log(1 - self.predict(self.X[i])))**2
        return J/len(self.y)

    def update_weights(self):
        y_pred = np.array([self.predict(i) for i in self.X])
        m = len(self.y)

        dw = np.dot(self.X, (y_pred - self.y).T)/m
        db = np.sum(y_pred - self.y)/m

        return self.weights - (self.learning_rate * dw), self.biases - (self.learning_rate * db) 
    
    def train(self):
        cost_his = []
        for _ in range(self.iteration):
            self.weights, self.biases = self.update_weights()
            cost = self.cost_function()
            cost_his.append(cost)

        return self.weights, self.biases, cost_his

    def visualize(self):
        plt.scatter(self.X, self.y, color='red')
        plt.plot(self.X, self.predict(self.X), color='blue')
        plt.title('Study hours vs Passing exam')
        plt.xlabel('Study hours')
        plt.ylabel('Pass = 1, Fail = 0')
        plt.show()
    
    
if __name__ == "__main__":
    X = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])
    y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
    clf = LogisticRegression(X, y, 0.5, 0, 0.01, 50000)
    weights, biases, _ = clf.train()
    print("f(x) = exp({} . x + {})".format(weights, biases))
    clf.visualize()