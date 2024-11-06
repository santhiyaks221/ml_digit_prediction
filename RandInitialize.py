import numpy as np

def initialise(a, b):
    epsilon = 0.15
    return np.random.rand(a, b + 1) * (2 * epsilon) - epsilon  # Randomly initializes values of thetas
