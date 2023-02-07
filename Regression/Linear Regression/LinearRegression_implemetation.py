import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_bar = sum(X)/len(X)
        self.y_bar = sum(y)/len(y)
    
    
    def centroid(self):
         return ("Centroid: ({}, {})".format(np.round(self.X_bar, 3), np.round(self.y_bar,3)))
    
    def coef(self):
        top = 0.0
        bot = 0.0
        for i in range(len(self.X)):
            top += (self.X[i] - self.X_bar) * (self.y[i] - self.y_bar)
            bot += (self.X[i] - self.X_bar) ** 2
        return top/bot
    
    def intercept(self):
        return self.y_bar - self.coef()* self.X_bar

    def predict(self, x):
        return self.coef()*x + self.intercept()

    def visualization(self):
        plt.scatter(self.X, self.y, color="red")
        plt.plot(self.X, self.coef() * self.X + self.intercept(), color="blue")
        plt.title("Height vs. Weight")
        plt.xlabel("Height")
        plt.ylabel("Weight")
        plt.show()

    def lost_function(self):
        L = 0.0
        for i in range(len(X)):
            L += (self.y[i] - (self.X[i] * self.coef() + self.intercept()))**2
        return 1/2 * L

if __name__ == "__main__":
    X = np.array([147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183])
    y = np.array([49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68])
    print(LinearRegression(X,y).centroid())
    print("b1 =", LinearRegression(X,y).coef())
    print("b0 =", LinearRegression(X,y).intercept())
    print("predict: height = 200 cm -> weight =", LinearRegression(X,y).predict(200))
    print("calculate error: e =", LinearRegression(X,y).lost_function())
    LinearRegression(X,y).visualization()





    