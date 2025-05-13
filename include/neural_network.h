#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10

// Structure for the neural network
typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    double* weights_input_hidden;
    double* weights_hidden_output;
    double* bias_hidden;
    double* bias_output;

    // DRTP-Specific buffers
    double* hidden_output;
    double* output_activation;
    double* random_targets;
} NeuralNetwork;

// Core Network functions
NeuralNetwork* CreateNeuralNetwork(int input_size, int hidden_size, int output_size);
void FreeNeuralNetwork(NeuralNetwork* nn);

// Forward and backwards pass
int ForwardPass(NeuralNetwork* nn, double* input, double* output);
void DRTPBackwardsPass(NeuralNetwork* nn, double* input, double* target);

// DRTP-specific functions
int GenerateRandomTargets(NeuralNetwork* nn, double* target);
int UpdateWeights(NeuralNetwork* nn, double* input, double* target);

// Activation Functions
double Sigmoid(double x);
double SigmoidDerivative(double x);

// Training and evaluation
int TrainingBatch(NeuralNetwork* nn, double* inputs, double* targets, int batch_size);
double EvaluateAccuracy(NeuralNetwork* nn, double* inputs, double* targets, int num_samples);

#endif // NEURAL_NETWORK_H 