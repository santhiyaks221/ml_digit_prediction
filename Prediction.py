import numpy as np

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))  # Adding bias unit to first layer
    z2 = np.dot(X, Theta1.T)
    a2 = 1 / (1 + np.exp(-z2))
    a2 = np.hstack((np.ones((m, 1)), a2))  # Adding bias unit
    z3 = np.dot(a2, Theta2.T)
    a3 = 1 / (1 + np.exp(-z3))
    return np.argmax(a3, axis=1)  # Predict the class based on maximum value
