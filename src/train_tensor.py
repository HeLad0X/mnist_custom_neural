import tensorflow as tf
import keras_tuner as kt
import numpy as np
from utils import read_mnist_images
from sklearn.metrics import classification_report

# Read and prepare the MNIST dataset
# Assuming X_train and X_test are already transposed and flattened
X_train, Y_train = read_mnist_images()
X_test, Y_test = read_mnist_images('test')

# Convert CuPy arrays to NumPy arrays (if they are CuPy arrays)
X_train = X_train.get()
Y_train = Y_train.get()
X_test = X_test.get()
Y_test = Y_test.get()

# Reshape the flattened and transposed images back to their original shape
X_train_reshaped = X_train.T.reshape(-1, 28, 28, 1)
X_test_reshaped = X_test.T.reshape(-1, 28, 28, 1)

# Define the hypermodel for hyperparameter tuning
def build_model(hp):
    model = tf.keras.Sequential([
        # Convolutional layer with adjustable number of filters and kernel size
        tf.keras.layers.Conv2D(
            filters=hp.Int('filters', min_value=32, max_value=128, step=32),
            kernel_size=hp.Choice('kernel_size', values=[3, 5]),
            activation='relu',
            input_shape=(28, 28, 1)),
        # Max pooling layer
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten layer to convert 2D to 1D
        tf.keras.layers.Flatten(),
        # Dense layer with an adjustable number of units
        tf.keras.layers.Dense(
            units=hp.Int('units', min_value=32, max_value=128, step=32),
            activation='relu'),
        # Output layer with 10 units for the 10 classes
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    # Compile the model with an adjustable learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[0.1, 1e-2, 1e-3])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model

# Set up the hyperparameter tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='hyperparam_tuning',
    project_name='mnist_cnn')

# Start the hyperparameter search
tuner.search(X_train_reshaped, Y_train, epochs=10, validation_split=0.2)


# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model
test_loss, test_accuracy = best_model.evaluate(X_test_reshaped, Y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

# Get predictions
predictions = best_model.predict(X_test_reshaped)

# Other evaluation metrics (e.g., classification report)
predicted_classes = np.argmax(predictions, axis=1)
print(classification_report(Y_test, predicted_classes))
