from src.train import get_predictions, load_model
from src.utils import  read_mnist_images
import cupy as cp

def compute_accuracy(predictions, labels):
    correct_predictions = cp.sum(predictions == labels)
    accuracy = correct_predictions / labels.shape[0] * 100
    return accuracy


def test_random_samples(X_test, Y_test, W1, b1, W2, b2, num_samples=5):
    # Choose random indices
    random_indices = cp.random.choice(X_test.shape[1], num_samples, replace=False)
    
    # Get the random samples
    X_random = X_test[:, random_indices]
    Y_random = Y_test[random_indices]
    
    # Generate predictions
    predictions = get_predictions(X_random, W1, b1, W2, b2)
    
    # Display results
    for i in range(num_samples):
        print(f"Sample {i + 1}:")
        print(f"Predicted: {predictions[i]}, Actual: {Y_random[i]}")
        print()


def evaluate(test_images, test_labels):
    best_params = load_model()
    print(test_images.shape)
    print(test_labels.shape)
    
    W1_best = best_params['W1']
    b1_best = best_params['b1']
    W2_best = best_params['W2']
    b2_best = best_params['b2']

    predictions = get_predictions(test_images, W1_best, b1_best, W2_best, b2_best)
    accuracy = compute_accuracy(predictions, test_labels)

    print('Predicting some random test cases')
    test_random_samples(test_images, test_labels, W1_best, b1_best, W2_best, b2_best, num_samples=10)

    print(f'Accuracy of the model is: {accuracy}')

if __name__ == '__main__':
    test_images, test_labels = read_mnist_images('test')
    evaluate(test_images, test_labels)