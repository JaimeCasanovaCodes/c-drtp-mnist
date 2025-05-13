#include "drtp.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdio.h>

#define LearningRate  0.3     
#define TargetScale   1.0     
#define MaxError      200.0    

// Structure to store fixed random projection weights
typedef struct {
    double* random_weights;  // Fixed random weights for target projection
} DRTPBuffers;

static DRTPBuffers buffers = {NULL};
static int sample_count = 0;  // Track samples for periodic reporting

// Sigmoid derivative helper function
static double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

void AllocateDRTPBuffers(NeuralNetwork* nn) {
    // Allocate buffer for random projection weights if not already done
    if(buffers.random_weights == NULL) {
        buffers.random_weights = (double*)malloc(nn->output_size * nn->hidden_size * sizeof(double));
        if(buffers.random_weights == NULL) return;
        
        // Initialize fixed random weights with larger scale
        srand(42);  // Fixed seed for reproducibility
        for(int i = 0; i < nn->output_size * nn->hidden_size; i++) {
            buffers.random_weights[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;  // Range [-1, 1]
        }
    }
    
    // Allocate network buffers
    nn->hidden_output = (double*)malloc(nn->hidden_size * sizeof(double));
    nn->output_activation = (double*)malloc(nn->output_size * sizeof(double));
    nn->random_targets = (double*)malloc(nn->hidden_size * sizeof(double));  // Now stores projected targets
}

void FreeDRTPBuffers(NeuralNetwork* nn) {
    if(nn == NULL) return;
    
    if(nn->hidden_output != NULL) free(nn->hidden_output);
    if(nn->output_activation != NULL) free(nn->output_activation);
    if(nn->random_targets != NULL) free(nn->random_targets);
    
    if(buffers.random_weights != NULL) {
        free(buffers.random_weights);
        buffers.random_weights = NULL;
    }
}

int GenerateRandomTargets(NeuralNetwork* nn, double* target) {
    if(nn == NULL || target == NULL || nn->random_targets == NULL) {
        printf("ERROR: Failed to generate random targets - null pointer\n");
        return -1;
    }
    
    // Project one-hot target through fixed random weights
    for(int i = 0; i < nn->hidden_size; i++) {
        double projection = 0.0;
        for(int j = 0; j < nn->output_size; j++) {
            projection += target[j] * buffers.random_weights[j * nn->hidden_size + i];
        }
        // Scale the projections for better learning
        nn->random_targets[i] = projection * TargetScale;
    }
    
    return 0;
}

int UpdateWeights(NeuralNetwork* nn, double* input, double* target) {
    if(nn == NULL || input == NULL || target == NULL) {
        printf("ERROR: Failed to update weights - null pointer\n");
        return -1;
    }
    
    double total_error = 0.0;
    
    // Update input-to-hidden weights using random target projections
    for(int i = 0; i < nn->hidden_size; i++) {
        double error = nn->random_targets[i] - nn->hidden_output[i];
        double delta = error * sigmoid_derivative(nn->hidden_output[i]);
        total_error += fabs(error);
        
        // Clip delta to prevent large updates
        if(delta > 0.1) delta = 0.1;
        if(delta < -0.1) delta = -0.1;
        
        for(int j = 0; j < nn->input_size; j++) {
            int idx = i * nn->input_size + j;
            nn->weights_input_hidden[idx] += LearningRate * delta * input[j];
            
            // Clip weights to prevent extreme values
            if(nn->weights_input_hidden[idx] > 2.0) nn->weights_input_hidden[idx] = 2.0;
            if(nn->weights_input_hidden[idx] < -2.0) nn->weights_input_hidden[idx] = -2.0;
        }
        
        nn->bias_hidden[i] += LearningRate * delta;
        if(nn->bias_hidden[i] > 2.0) nn->bias_hidden[i] = 2.0;
        if(nn->bias_hidden[i] < -2.0) nn->bias_hidden[i] = -2.0;
    }
    
    // Update hidden-to-output weights using actual targets
    for(int i = 0; i < nn->output_size; i++) {
        double error = target[i] - nn->output_activation[i];
        double delta = error * sigmoid_derivative(nn->output_activation[i]);
        total_error += fabs(error);
        
        // Clip delta to prevent large updates
        if(delta > 0.1) delta = 0.1;
        if(delta < -0.1) delta = -0.1;
        
        for(int j = 0; j < nn->hidden_size; j++) {
            int idx = i * nn->hidden_size + j;
            nn->weights_hidden_output[idx] += LearningRate * delta * nn->hidden_output[j];
            
            // Clip weights to prevent extreme values
            if(nn->weights_hidden_output[idx] > 2.0) nn->weights_hidden_output[idx] = 2.0;
            if(nn->weights_hidden_output[idx] < -2.0) nn->weights_hidden_output[idx] = -2.0;
        }
        
        nn->bias_output[i] += LearningRate * delta;
        if(nn->bias_output[i] > 2.0) nn->bias_output[i] = 2.0;
        if(nn->bias_output[i] < -2.0) nn->bias_output[i] = -2.0;
    }
    
    // Print error every 5000 samples you can change this to what you want
    sample_count++;
    if(sample_count % 5000 == 0) {
        printf("Progress: %d samples - Average Error: %.6f\n", sample_count, total_error / (nn->hidden_size + nn->output_size));
    }
    
    // Check for training failure (error too high)
    if(total_error > MaxError) {
        printf("TRAINING FAILURE at sample %d - Error: %.6f\n", sample_count, total_error);
        return -1;
    }
    
    return 0;
} 