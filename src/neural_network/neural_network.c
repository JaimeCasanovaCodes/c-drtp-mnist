#include <stdlib.h>
#include <math.h>
#include "neural_network.h"
#include <time.h>
#include <stdio.h>
#include "initialization.h"
#include "drtp.h"

NeuralNetwork* CreateNeuralNetwork(int input_size, int hidden_size, int output_size) {
    if(input_size < 0 && hidden_size < 0 && output_size < 0) {
        printf("Sizes are Negative must be Positive\n");
        return NULL;
    }

    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if(nn == NULL) {
        return NULL;
    }

    nn->input_size = input_size;
    nn->output_size = output_size;
    nn->hidden_size = hidden_size;

    // Weights from input to hidden
    nn->weights_input_hidden = (double*)malloc(input_size * hidden_size * sizeof(double));
    if(nn->weights_input_hidden == NULL) {
        free(nn);
        return NULL;
    }

    // Weights from hidden to output
    nn->weights_hidden_output = (double*)malloc(hidden_size * output_size * sizeof(double));
    if(nn->weights_hidden_output == NULL) {
        free(nn->weights_input_hidden);
        free(nn);
        return NULL;
    }

    // Bias from hidden
    nn->bias_hidden = (double*)malloc(hidden_size * sizeof(double));
    if(nn->bias_hidden == NULL) {
        free(nn->weights_hidden_output);
        free(nn->weights_input_hidden);
        free(nn);
        return NULL;
    }

    nn->bias_output = (double*)malloc(output_size * sizeof(double));
    if(nn->bias_output == NULL) {
        free(nn->bias_hidden);
        free(nn->weights_hidden_output);
        free(nn->weights_input_hidden);
        free(nn);
        return NULL;
    }

    // Initialize weights and biases
    InitializeWeights(nn);
    InitializeBiases(nn);

    // Allocate DRTP buffers
    AllocateDRTPBuffers(nn);
    if(nn->random_targets == NULL) {
        FreeNeuralNetwork(nn);
        return NULL;
    }

    return nn;
}

void FreeNeuralNetwork(NeuralNetwork* nn) {
    if(nn == NULL) {
        return;
    }

    // Free all allocated memory in reverse order
    if(nn->random_targets != NULL) {
        free(nn->random_targets);
    }
    if(nn->output_activation != NULL) {
        free(nn->output_activation);
    }
    if(nn->hidden_output != NULL) {
        free(nn->hidden_output);
    }
    if(nn->bias_output != NULL) {
        free(nn->bias_output);
    }
    if(nn->bias_hidden != NULL) {
        free(nn->bias_hidden);
    }
    if(nn->weights_hidden_output != NULL) {
        free(nn->weights_hidden_output);
    }
    if(nn->weights_input_hidden != NULL) {
        free(nn->weights_input_hidden);
    }
    free(nn);
}
