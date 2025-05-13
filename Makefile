CC = gcc
CFLAGS = -Wall -Wextra -O2 -I./src -I./include
LDFLAGS = -lm

SRCS = src/main.c \
       src/neural_network/neural_network.c \
       src/mnist/mnist_csv.c \
       src/activation/activation.c \
       src/training/initialization.c \
       src/training/training.c \
       src/drtp/drtp.c \
       src/forward_pass/forward_pass.c

OBJS = $(SRCS:.c=.o)
TARGET = drtp_nn

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	del /F /Q src\main.o src\neural_network\neural_network.o src\mnist\mnist_csv.o src\activation\activation.o src\training\initialization.o src\training\training.o src\drtp\drtp.o src\forward_pass\forward_pass.o drtp_nn.exe

.PHONY: all clean 