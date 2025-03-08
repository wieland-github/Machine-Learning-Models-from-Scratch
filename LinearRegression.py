import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iterations):
            y_prediction = np.dot(X, self.weights) + self.bias 

            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_prediction - y))
            db = (1/n_samples) * np.sum(y_prediction - y)

            # Update weights
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X): 
        y_prediction = np.dot(X, self.weights) + self.bias
        return y_prediction
