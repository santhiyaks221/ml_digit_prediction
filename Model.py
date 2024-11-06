import numpy as np

def neural_network(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], 
                        (num_labels, hidden_layer_size + 1))

    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))  # Adding bias unit to first layer

    # Forward propagation
    z2 = np.dot(X, Theta1.T)
    a2 = 1 / (1 + np.exp(-z2))
    a2 = np.hstack((np.ones((m, 1)), a2))  # Adding bias unit
    z3 = np.dot(a2, Theta2.T)
    a3 = 1 / (1 + np.exp(-z3))

    # Create y vector
    y_vect = np.eye(num_labels)[y.astype(int)]  # One-hot encoding

    # Cost function
    J = (1 / m) * np.sum(-y_vect * np.log(a3) - (1 - y_vect) * np.log(1 - a3)) + \
        (lamb / (2 * m)) * (np.sum(np.square(Theta1[:, 1:])) + np.sum(np.square(Theta2[:, 1:])))

    # Backpropagation
    Delta3 = a3 - y_vect
    Delta2 = np.dot(Delta3, Theta2) * a2 * (1 - a2)
    Delta2 = Delta2[:, 1:]  # Remove bias unit

    # Gradients
    Theta1_grad = (1 / m) * np.dot(Delta2.T, X) + (lamb / m) * np.hstack((np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:]))
    Theta2_grad = (1 / m) * np.dot(Delta3.T, a2) + (lamb / m) * np.hstack((np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:]))

    return J, np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))
