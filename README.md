### MNIST Neural Network in C

This project implements a neural network using Direct Random Target Projection (DRTP) in C for recognizing handwritten digits from the MNIST dataset. This is one of the first implementations of DRTP in C (or first from what I have seen online), offering potential performance advantages over higher-level language implementations. I'm doing this for educational purposes I do not have any type of research background. I am a undergraduate college student.

## Direct Random Target Projection

This is an alternative nueral network training algorithm that is different from backpropogation in several ways. It projects random targets directly onto the hidden layers instead of propagating errors backwards through the network. These targets are used to guide the learning process. It is a simpler implementation compared to backpropagation. Can be more efficient for computing. Does not require storing gradients for all layers. Can work well with non-dofferentiable activation functions such as Sigmoid.

## Project Structure

```
src/
├── activation/         
│   ├── activation.h    # Activation function declarations
│   └── activation.c    # Sigmoid and derivative implementations
├── drtp/              
│   ├── drtp.h         # DRTP function declarations
│   └── drtp.c         # DRTP implementation
├── forward_pass/      
│   ├── forward_pass.h # Forward pass declarations
│   └── forward_pass.c # Forward propagation implementation
├── neural_network/    
│   ├── neural_network.h # Core network structure and declarations
│   └── neural_network.c # Network creation and memory management
├── training/          
│   ├── training.h     # Training function declarations
│   ├── training.c     # Training implementation
│   ├── initialization.h # Weight/bias initialization declarations
│   └── initialization.c # Weight/bias initialization implementation
├── mnist/            
│   ├── mnist_csv.h        # MNIST data structure and declarations
│   └── mnist_csv.c        # MNIST data loading and processing
├── main.c            # Main program entry point
└── Makefile          # Build configuration
```

## Requirements

- GCC compiler
- Make
- MNIST dataset files (train-images-idx3-ubyte and train-labels-idx1-ubyte) I put the csv version in data the originals were giving me headaches (I was only able to put test so heres the link to both test and train - https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

## Building the Project

```bash
make
```

## Cleaning Build Files

```bash
make clean
```

## Usage

After building, run the program:

```bash
./drtp_nn
```

## Implementation Details

The neural network consists of:
- Input layer (784 neurons for 28x28 pixel images)
- Hidden layer (256 neurons can change if you want)
- Output layer (10 neurons for digits 0-9)

## Results
So far from what I have been testing:
 - Training (Highest Accuracy):
 - Testing (Highest Accuracy): 

## License

This project is open source and available under the MIT License. 