#include "forward_pass.h"
#include <math.h>
#include <stdio.h>
#include <stddef.h>

// Forward pass implementation
int ForwardPass(NeuralNetwork* nn, double* input, double* output) {
    if(nn == NULL || input == NULL || output == NULL) {
        printf("ForwardPass: Null pointer detected\n");
        return -1;
    }

    // Check for Not a number (NaN) or Inf in input
    for(int i = 0; i < nn->input_size; i++) {
        if(isnan(input[i]) || isinf(input[i])) {
            printf("ForwardPass: Invalid input value at index %d: %f\n", i, input[i]);
            return -2;
        }
    }

    // Hidden Layer calculations
    for(int i = 0; i < nn->hidden_size; i++) {
        double result = 0;

        // input * the hidden weight to give the result
        for(int j = 0; j < nn->input_size; j++) {
            result += input[j] * nn->weights_input_hidden[i * nn->input_size + j];
        }

        // Bias hidden added to Result
        result += nn->bias_hidden[i];
        
        // Check for NaN or Inf before sigmoid
        if(isnan(result) || isinf(result)) {
            printf("ForwardPass: Invalid hidden layer value before sigmoid at index %d: %f\n", i, result);
            return -3;
        }

        // Sigmoid Result
        result = Sigmoid(result);
        nn->hidden_output[i] = result;

        // Check for NaN or Inf after sigmoid
        if(isnan(result) || isinf(result)) {
            printf("ForwardPass: Invalid hidden layer value after sigmoid at index %d: %f\n", i, result);
            return -4;
        }
    }

    // Output Layer calculations
    for(int i = 0; i < nn->output_size; i++) {
        double result = 0;

        // hidden * weighted output to get result
        for(int j = 0; j < nn->hidden_size; j++) {
            result += nn->hidden_output[j] * nn->weights_hidden_output[i * nn->hidden_size + j];
        }

        // Bias output added to result
        result += nn->bias_output[i];

        // Check for NaN or Inf before sigmoid
        if(isnan(result) || isinf(result)) {
            printf("ForwardPass: Invalid output layer value before sigmoid at index %d: %f\n", i, result);
            return -5;
        }

        // Sigmoid Result
        result = Sigmoid(result);
        output[i] = result;

        // Check for NaN or Inf after sigmoid
        if(isnan(result) || isinf(result)) {
            printf("ForwardPass: Invalid output layer value after sigmoid at index %d: %f\n", i, result);
            return -6;
        }
    }
    return 0;
} 