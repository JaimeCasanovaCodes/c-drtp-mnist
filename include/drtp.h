#ifndef DRTP_H
#define DRTP_H

#include "neural_network.h"

// DRTP-specific function declarations
void AllocateDRTPBuffers(NeuralNetwork* nn);
int GenerateRandomTargets(NeuralNetwork* nn, double* target);
int UpdateWeights(NeuralNetwork* nn, double* input, double* target);

#endif // DRTP_H 