from datasets import load_dataset
import numpy as np
from Model import neural_network
from RandInitialize import initialise
from Prediction import predict
from scipy.optimize import minimize

# Load the MNIST dataset
dataset = load_dataset("ylecun/mnist")

# Extract training and testing data
X_train = dataset['train']['image']
y_train = dataset['train']['label']
X_test = dataset['test']['image']
y_test = dataset['test']['label']

# Preprocess the images and labels
X_train = np.array([np.array(img).flatten() / 255.0 for img in X_train])  # Normalize and flatten
X_test = np.array([np.array(img).flatten() / 255.0 for img in X_test])    # Normalize and flatten
y_train = np.array(y_train)
y_test = np.array(y_test)

# Neural Network parameters
input_layer_size = 784  # 28*28
hidden_layer_size = 100
num_labels = 10

# Randomly initializing Thetas
initial_Theta1 = initialise(hidden_layer_size, input_layer_size)
initial_Theta2 = initialise(num_labels, hidden_layer_size)

# Unroll parameters into a single column vector
initial_nn_params = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))

# Regularization parameter
lambda_reg = 0.1
maxiter = 70

# Training the neural network
myargs = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda_reg)
results = minimize(neural_network, initial_nn_params, args=myargs, 
                   options={'disp': True, 'maxiter': maxiter}, method="L-BFGS-B", jac=True)

nn_params = results["x"]
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1))
Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1))

# Evaluate accuracy on training and test sets
train_pred = predict(Theta1, Theta2, X_train)
test_pred = predict(Theta1, Theta2, X_test)

print('Training Set Accuracy: {:.4f}%'.format(np.mean(train_pred == y_train) * 100))
print('Test Set Accuracy: {:.4f}%'.format(np.mean(test_pred == y_test) * 100))

# Save the trained parameters
np.savetxt('Theta1.txt', Theta1, delimiter=' ')
np.savetxt('Theta2.txt', Theta2, delimiter=' ')
