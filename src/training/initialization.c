#include "initialization.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Simple random initialization for DRTP
void InitializeWeights(NeuralNetwork* nn) {
    // Use a better random seed
    srand((unsigned int)time(NULL) ^ (unsigned int)nn);

    // Initialize input-to-hidden weights with small random values
    for(int i = 0; i < (nn->input_size * nn->hidden_size); i++) {
        nn->weights_input_hidden[i] = ((double)rand() / RAND_MAX) * 0.4 - 0.2; // Range: [-0.2, 0.2]
    }
    
    // Initialize hidden-to-output weights with small random values
    for(int i = 0; i < (nn->hidden_size * nn->output_size); i++) {
        nn->weights_hidden_output[i] = ((double)rand() / RAND_MAX) * 0.4 - 0.2; // Range: [-0.2, 0.2]
    }
}

void InitializeBiases(NeuralNetwork* nn) {
    // Initialize biases to small values
    double bias_scale = 0.2;  // Increased bias scale for better initialization
    
    // Hidden Layer Bias
    for(int i = 0; i < nn->hidden_size; i++) {
        nn->bias_hidden[i] = ((double)rand() / RAND_MAX) * 2.0 * bias_scale - bias_scale;
    }
    
    // Output Layer Bias
    for(int i = 0; i < nn->output_size; i++) {
        nn->bias_output[i] = ((double)rand() / RAND_MAX) * 2.0 * bias_scale - bias_scale;
    }
} 