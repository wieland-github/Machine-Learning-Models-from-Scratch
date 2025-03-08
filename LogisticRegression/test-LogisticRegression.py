import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from LogisticRegression import LogisticRegression

breast_cancer = datasets.load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

logReg = LogisticRegression()
logReg.fit(X_train, y_train)
y_pred = logReg.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_pred==y_test)/len(y_test)

accuracy = accuracy(y_test, y_pred)
print(f"The accuracy is {accuracy}")

# Example test patient
new_patient = np.array([[
    17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]]) 

# Make prediction for a sample Patient
prediction = logReg.predict(new_patient)

# Interpret the result
if prediction[0] == 1:
    print("The model predicts that the patient does NOT have breast cancer (Benign).")
else:
    print("The model predicts that the patient HAS breast cancer (Malignant).")
