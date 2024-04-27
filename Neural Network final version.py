import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load training and test data
data = pd.read_csv("D:/Nunmbers_recognition(neural network)/train_data/train.csv") #you might wanna change the path based on where u have put the datasets
data_test = pd.read_csv("D:/Nunmbers_recognition(neural network)/test_data/test.csv")

# Convert data to numpy arrays
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # shuffle data for randomness

test = np.array(data)
m1, n1 = test.shape

# Prepare training data
data_train = data[0:m].T
Y_train = data_train[0]  # Labels
X_train = data_train[1:n]  # Features
X_train = X_train / 255.  # Normalize pixel values

# Prepare test data
data_test = test[0:m1].T
X_test = data_test[1:n1] # Features
X_test = X_test / 255. # Normalize pixel values

# Initialize parameters randomly
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

# Activation functions
def ReLU(Z):
    return np.maximum(Z, 0)

def tanh(Z):
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# Forward propagation
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = sigmoid(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Derivatives of activation functions
def ReLU_deriv(Z):
    return Z > 0

def tanh_deriv(Z):
    sech_Z = 2 / (np.exp(Z) + np.exp(-Z))
    return sech_Z**2

def sigmoid_deriv(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

# Convert labels to one-hot encoding
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Backward propagation
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * sigmoid_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

# Update parameters using gradient descent
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

# Get predictions from output probabilities
def get_predictions(A2):
    return np.argmax(A2, 0)

# Calculate accuracy of predictions
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Gradient descent algorithm
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    accuracy_history = []
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
            accuracy = get_accuracy(predictions, Y)
            accuracy_history.append(accuracy)
    return W1, b1, W2, b2, accuracy_history

nb_image_train = 5000

# Run gradient descent to train the model
W1, b1, W2, b2, accuracy_history = gradient_descent(X_train, Y_train, 0.1, nb_image_train)

# Make predictions on test data
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# Display a single test prediction along with its label
def test_prediction(index, W1, b1, W2, b2):
    if index < len(X_test[0]):
        current_image = X_test[:, index, None]
        prediction = make_predictions(X_test[:, index, None], W1, b1, W2, b2)
        print("Prediction: ", prediction)
        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()
    else:
        print("Index out of range for test data.")

# Plot the development of accuracy during training
def draw_development(accuracy_history):
    plt.plot(accuracy_history)
    plt.title('Accuracy Development')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.show()

# Draw accuracy development during training
draw_development(accuracy_history)

# Test a specific prediction
test_prediction(55, W1, b1, W2, b2)
