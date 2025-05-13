#include "training.h"
#include "forward_pass.h"
#include "drtp.h"
#include <stddef.h>
#include <stdio.h>

// Training implementation
int TrainingBatch(NeuralNetwork* nn, double* inputs, double* targets, int batch_size) {
    if(nn == NULL || inputs == NULL || targets == NULL || batch_size <= 0) {
        printf("Training batch is empty\n");
        return -1;
    }
    
    for(int i = 0; i < batch_size; i++) {
        double* current_input = inputs;
        double* current_target = targets;

        // Forward pass
        int forward_result = ForwardPass(nn, current_input, nn->output_activation);
        if(forward_result != 0) {
            printf("Error: ForwardPass failed on sample %d\n", i);
            return -2;
        }

        // Generate random targets
        int random_target_result = GenerateRandomTargets(nn, current_target);
        if(random_target_result != 0) {
            printf("Error: GenerateRandomTargets failed on sample %d\n", i);
            return -3;
        }

        // Update weights
        int update_result = UpdateWeights(nn, current_input, current_target);
        if(update_result != 0) {
            printf("Error: UpdateWeights failed on sample %d\n", i);
            return -4;
        }

        // Move to next sample
        inputs += 784;  // Move to next image
        targets += 10;  // Move to next target
    }
    return 0;
}

double EvaluateAccuracy(NeuralNetwork* nn, double* inputs, double* targets, int num_samples) {
    if(nn == NULL || inputs == NULL || targets == NULL || num_samples <= 0) {
        printf("Error: Invalid parameters in EvaluateAccuracy\n");
        return -1.0;
    }

    int correct = 0;

    for(int i = 0; i < num_samples; i++) {
        double* current_sample = inputs + (i * 784);  // Use MNIST_IMAGE_SIZE (784) directly
        double* current_target = targets + (i * nn->output_size);
        int sample_correct = 1;  // Assume sample is correct until proven wrong

        if(ForwardPass(nn, current_sample, nn->output_activation) != 0) {
            printf("Error: ForwardPass failed during accuracy evaluation on sample %d\n", i);
            return -5;
        }

        for(int j = 0; j < nn->output_size; j++) {
            if((nn->output_activation[j] > 0.5 && current_target[j] != 1) ||
               (nn->output_activation[j] < 0.5 && current_target[j] != 0)) {
                sample_correct = 0;  // Sample is incorrect if any output is wrong
                break;
            }
        }
        correct += sample_correct;
    }

    return (double)correct / num_samples;
} 