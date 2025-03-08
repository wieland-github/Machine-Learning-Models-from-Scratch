import numpy as np
import pandas as pd
from DecisionTree import DecisionTree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Breast Cancer Dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Combine data and labels for compatibility with custom DecisionTree class
dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)

# Split data into training and test sets
train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=42)

# Instantiate and train the DecisionTree model
dt = DecisionTree(max_depth=2, min_samples=2)
dt.fit(train_data[:, :-1], train_data[:, -1])

# Predictions on the test set
y_pred = dt.predict(test_data[:, :-1])

# Calculate accuracy
accuracy = accuracy_score(test_data[:, -1], y_pred)
print(f"Test Accuracy: {accuracy:.2f}")


