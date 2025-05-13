#ifndef MNIST_CSV_H
#define MNIST_CSV_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double* images;      // Flattened array of images (each image is 28x28 = 784 pixels)
    int* labels;        // Array of labels (0-9)
    int num_images;     // Number of images
    int image_size;     // Size of each image (28x28 = 784)
} MNISTData;

// Load MNIST data from CSV files
MNISTData* load_mnist_csv(const char* images_path, const char* labels_path);

// Free MNIST data
void free_mnist_data(MNISTData* data);

#endif // MNIST_CSV_H 