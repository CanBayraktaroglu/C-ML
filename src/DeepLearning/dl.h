#ifndef __DL_H__
#define __DL_H__

#include <stdlib.h>	
#include <string.h>
#include "utils.h"
#include "matrix.h"
#include <math.h>

typedef struct {
    double weight;
    double bias;
} Neuron;

typedef struct {
    size_t num_neurons;
    double (*act_fn)(double x);
    Matrix* (*forward_pass)(Matrix* X);
    Matrix* weights;
    Matrix* biases;
    Neuron* neurons;
}FeedForwardLayer;

// ACTIVATION FUNCTIONS
double relu(double x){
    if (x > 0) return x;
    return 0;
};

double sigmoid(double x){
    return 1/(1 + exp(-x));
};

double tanh(double x){
    return exp(2*x - 1)/exp(2*x + 1);
};

Matrix* feed_forward_forward_pass(FeedForwardLayer* layer ,Matrix* X){
    return NULL;
};












#endif // __DL_H__