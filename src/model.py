import cupy as cp
from src.utils import (
                    init_params, 
                    ReLU, 
                    softmax, 
                    one_hot, 
                    ReLU_derivative, 
                    compute_cost
                   )
from src.config import ParamConfig

def forward_propagation(X, W1, b1, W2, b2):
    # Linear combination for the first layer
    Z1 = W1.dot(X) + b1
    
    # Activation using ReLU for the first layer
    A1 = ReLU(Z1)
    
    # Linear combination for the second layer (output layer)
    Z2 = W2.dot(A1) + b2
    
    # Activation using softmax for the output layer to get probabilities
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2


def backward_propagation(X, Y, A1, A2, W2, Z1, m):
    # Derivative of the loss function with respect to Z2
    dZ2 = A2 - one_hot(Y)
    
    # Gradients for W2 and b2
    dW2 = (1 / m) * cp.dot(dZ2, A1.T)  # Adjusted to match shapes
    db2 = (1 / m) * cp.sum(dZ2, axis=1, keepdims=True)  # Sum over correct axis
    
    # Derivative of the activation function for layer 1
    dA1 = cp.dot(W2.T, dZ2)  # Adjusted for transpose
    dZ1 = dA1 * ReLU_derivative(Z1)
    
    # Gradients for W1 and b1
    dW1 = (1 / m) * cp.dot(dZ1, X.T)  # Adjusted to match shapes
    db1 = (1 / m) * cp.sum(dZ1, axis=1, keepdims=True)  # Sum over correct axis
    
    return dW1, db1, dW2, db2


def gradient_descent(X_train, Y_train, W1, b1, W2, b2, learning_rate, epochs, batch_size):
    m = X_train.shape[1]  # number of training examples
    best_cost = 1
    best_W1 = None
    best_W2 = None
    best_b1 = None
    best_b2 = None

    for epoch in range(epochs):
        # Shuffle the data
        permutation = cp.random.permutation(m)
        X_shuffled = X_train[:, permutation]
        Y_shuffled = Y_train[permutation]

        for i in range(0, m, batch_size):
            # Get the next batch
            X_batch = X_shuffled[:, i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]

            # Forward propagation
            Z1, A1, Z2, A2 = forward_propagation(X_batch, W1, b1, W2, b2)

            # Backward propagation
            dW1, db1, dW2, db2 = backward_propagation(X_batch, Y_batch, A1, A2, W2, Z1, batch_size)

            # Update weights and biases
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

            if best_b1 is None:
                best_b1 = b1
                best_b2 = b2
                best_W1 = W1
                best_W2 = W2

        # Optionally: print the cost every few epochs for monitoring
        if (epoch) % 5 == 0 or epoch == epochs-1:
            _, _, _, A2 = forward_propagation(X_train, W1, b1, W2, b2)
            cost = compute_cost(A2, Y_train)
            if cost < best_cost:
                best_b1 = b1
                best_b2 = b2
                best_W1 = W1
                best_W2 = W2
            print(f"Epoch {epoch}, Cost: {cost}")


    return best_W1, best_b1, best_W2, best_b2

def hyperparameter_tuning(X_train, Y_train, W1, b1, W2, b2, learning_rates, epochs_list, batch_size):
    best_cost = float('inf')
    best_params = {}

    n=1
    for lr in learning_rates:
        for epochs in epochs_list:
            print(f'Parameter tuning iteration: {n}')
            n+=1
            # Initialize weights and biases for each trial
            W1_temp, b1_temp = W1.copy(), b1.copy()
            W2_temp, b2_temp = W2.copy(), b2.copy()
            
            # Train the model with cur[re]nt hyperparameters
            W1_temp, b1_temp, W2_temp, b2_temp = gradient_descent(X_train, Y_train, W1_temp, b1_temp, W2_temp, b2_temp, lr, epochs, batch_size)

            # Compute the cost with the current parameters
            _, _, _, A2 = forward_propagation(X_train, W1_temp, b1_temp, W2_temp, b2_temp)
            cost = compute_cost(A2, Y_train)

            # Save the best parameters
            if cost < best_cost:
                best_cost = cost
                best_params = {
                    'learning_rate': lr,
                    'epochs': epochs,
                    'W1': W1_temp,
                    'b1': b1_temp,
                    'W2': W2_temp,
                    'b2': b2_temp
                }
    
    print(f"Best cost: {best_cost}")
    print(f"Best learning rate: {best_params['learning_rate']}")
    print(f"Best epochs: {best_params['epochs']}")
    
    return best_params


def start_hyperparameter_tuning(image_train, label_train):
    # Initializing the weights and biases
    W1, b1, W2, b2 = init_params()

    # Start hyperparameter tuning
    best_params = hyperparameter_tuning(image_train, label_train, W1, b1, W2, b2, \
                                         ParamConfig.LEARNING_RATES, ParamConfig.EPOCHS, ParamConfig.BATCH_SIZE)

    return best_params


