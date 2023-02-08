import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class SVM:
    def __init__(self, X, y, w, b, lr = 0.001, lambda_params = 0.01, n_iter = 1000):
        self.X = X
        self.y = y
        self.lr = lr
        self.lambda_params = lambda_params
        self.n_iter = n_iter
        self.w = w
        self.b = b
    
    def train(self):
        for _ in range(self.n_iter):
            for i, X_i in enumerate(self.X):
                condition = self.y[i] * (np.dot(X_i, self.w) - self.b)
                if condition >= 1:
                    self.w -= self.lr * (2 * self.lambda_params * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_params * self.w - np.dot(X_i, self.y[i]))
                    self.b -= self.lr * self.y[i]
        return self.w, self.b
    
    def predict(self, var):
        return np.sign(np.dot(self.w, var) - self.b)
    
    def visualize_svm(self):
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = get_hyperplane_value(x0_1, self.w, self.b, 0)
        x1_2 = get_hyperplane_value(x0_2, self.w, self.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, self.w, self.b, -1)
        x1_2_m = get_hyperplane_value(x0_2, self.w, self.b, -1)

        x1_1_p = get_hyperplane_value(x0_1, self.w, self.b, 1)
        x1_2_p = get_hyperplane_value(x0_2, self.w, self.b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()

if __name__ == "__main__":
    X, y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
    )
    y = np.where(y == 0, -1, 1)
    # print(X, y)

    clf = SVM(X, y, w=np.zeros(X.shape[1]), b = 0)
    w, b = clf.train()
    print("f(x) = {} . x - ({})".format(w, b))

    clf.visualize_svm()
    