#include "activation.h"
#include <math.h>

// Sigmoid Formulas
double Sigmoid(double x) {
    double y = 1 / (1 + exp(-x));
    return y;
}

double SigmoidDerivative(double x) {
    // For sigmoid, f'(x) = f(x) * (1 - f(x))
    // where x is already the output of sigmoid
    return x * (1.0 - x);
} 