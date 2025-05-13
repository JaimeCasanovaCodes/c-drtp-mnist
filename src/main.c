#include <stdio.h>
#include <stdlib.h>
#include "neural_network.h"
#include "mnist_csv.h"
#include "activation.h"
#include "forward_pass.h"
#include "drtp.h"
#include "training.h"
#include <time.h>

// MNIST constants
#define MNIST_IMAGE_SIZE 784  // 28x28 pixels
#define MNIST_NUM_CLASSES 10  // Digits 0-9
#define BATCH_SIZE 100
#define NUM_ITERATIONS 5      // Number of Epochs

void printProgress(int current, int total) {
    float progress = (float)current / total * 100;
    printf("\rProgress: %.1f%% ", progress);
    fflush(stdout);
}

int main() {
    // Load MNIST data
    printf("Loading MNIST data...\n");
    MNISTData* train_data = load_mnist_csv("data/mnist_train.csv", NULL);  // Labels are in the same file
    if (!train_data) {
        printf("Failed to load MNIST training data\n");
        return 1;
    }

    printf("Data loaded successfully. Training set size: %d images\n", train_data->num_images);

    // Create one-hot encoded labels
    double* encoded_labels = (double*)malloc(train_data->num_images * MNIST_NUM_CLASSES * sizeof(double));
    if (!encoded_labels) {
        printf("Failed to allocate memory for encoded labels\n");
        free_mnist_data(train_data);
        return 1;
    }

    // One-hot encode the labels
    for (int i = 0; i < train_data->num_images; i++) {
        for (int j = 0; j < MNIST_NUM_CLASSES; j++) {
            encoded_labels[i * MNIST_NUM_CLASSES + j] = (train_data->labels[i] == j) ? 1.0 : 0.0;
        }
    }

    // Create neural network
    printf("Creating neural network...\n");
    NeuralNetwork* nn = CreateNeuralNetwork(MNIST_IMAGE_SIZE, HIDDEN_SIZE, MNIST_NUM_CLASSES);
    if (!nn) {
        printf("Failed to create neural network\n");
        free(encoded_labels);
        free_mnist_data(train_data);
        return 1;
    }

    // Training loop
    printf("Starting training...\n");
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        printf("\nIteration %d/%d\n", iter + 1, NUM_ITERATIONS);
        
        // Start timing this epoch
        clock_t start = clock();
        
        for (int i = 0; i < train_data->num_images; i += BATCH_SIZE) {
            int current_batch_size = (i + BATCH_SIZE <= train_data->num_images) ? BATCH_SIZE : (train_data->num_images - i);
            
            if (TrainingBatch(nn, 
                            &train_data->images[i * MNIST_IMAGE_SIZE],
                            &encoded_labels[i * MNIST_NUM_CLASSES],
                            current_batch_size) != 0) {
                printf("Training failed\n");
                break;
            }
            
            printProgress(i + current_batch_size, train_data->num_images);
        }
        
        // End timing and calculate duration
        clock_t end = clock();
        double epoch_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("\nEpoch %d completed in %.2f seconds\n", iter + 1, epoch_time);
    }

    // Evaluate final accuracy
    double accuracy = EvaluateAccuracy(nn, train_data->images, encoded_labels, train_data->num_images);
    printf("\nFinal Training Accuracy: %.2f%%\n", accuracy * 100);

    // Load MNIST test data
    printf("\nLoading MNIST test data...\n");
    MNISTData* test_data = load_mnist_csv("data/mnist_test.csv", NULL);
    if (!test_data) {
        printf("Failed to load MNIST test data\n");
        free(encoded_labels);
        free_mnist_data(train_data);
        FreeNeuralNetwork(nn);
        return 1;
    }

    printf("Test data loaded successfully. Test set size: %d images\n", test_data->num_images);

    // Create one-hot encoded labels for test data
    double* test_encoded_labels = (double*)malloc(test_data->num_images * MNIST_NUM_CLASSES * sizeof(double));
    if (!test_encoded_labels) {
        printf("Failed to allocate memory for test encoded labels\n");
        free_mnist_data(test_data);
        free(encoded_labels);
        free_mnist_data(train_data);
        FreeNeuralNetwork(nn);
        return 1;
    }

    // One-hot encode the test labels
    for (int i = 0; i < test_data->num_images; i++) {
        for (int j = 0; j < MNIST_NUM_CLASSES; j++) {
            test_encoded_labels[i * MNIST_NUM_CLASSES + j] = (test_data->labels[i] == j) ? 1.0 : 0.0;
        }
    }

    // Evaluate on test set
    printf("Evaluating on test set...\n");
    double test_accuracy = EvaluateAccuracy(nn, test_data->images, test_encoded_labels, test_data->num_images);
    printf("\nTest Set Accuracy: %.2f%%\n", test_accuracy * 100);

    // Cleanup test data
    free(test_encoded_labels);
    free_mnist_data(test_data);

    // Cleanup
    free(encoded_labels);
    free_mnist_data(train_data);
    FreeNeuralNetwork(nn);

    return 0;
}