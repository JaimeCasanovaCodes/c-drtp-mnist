#ifndef TRAINING_H
#define TRAINING_H

#include "neural_network.h"

// Training function declarations
int TrainingBatch(NeuralNetwork* nn, double* inputs, double* targets, int batch_size);
double EvaluateAccuracy(NeuralNetwork* nn, double* inputs, double* targets, int num_samples);

#endif // TRAINING_H 