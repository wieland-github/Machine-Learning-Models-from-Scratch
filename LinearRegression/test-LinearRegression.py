import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression  # Make sure this is correct!


X, y = datasets.make_regression(n_samples=100, n_features=1, noise=30, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Model
reg = LinearRegression()
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

# Compute Mean Squared Error
def mse(y_test, predictions):
    return np.mean((y_test - predictions) ** 2)

mse_value = mse(y_test, predictions)
print(f"The Mean Squared Error is {mse_value}")
 
y_pred_line = reg.predict(X)

# Plot Decision Boundary (Regression Line)
fig = plt.figure(figsize=(8,6))
plt.scatter(X_train, y_train, color="b", marker="o", s=30, label="Training Data")
plt.scatter(X_test, y_test, color="c", s=10, label="Test Data")
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.legend()
plt.show()
