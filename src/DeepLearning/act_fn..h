#ifndef ACT_FN_H_
#define ACT_FN_H

#include <stdlib.h>	
#include <string.h>
#include "matrix.h"
#include <math.h>

#pragma region Activation Functions

// ACTIVATION FUNCTIONS
void matrix_relu(Matrix* X){
    if (X->n_cols){        
        for (size_t i = 0; i < X->n_rows; i++){
            for (size_t j = 0; j < X->n_cols; j++){
                if (matrix_get(X, i, j) >= 0) continue;
                matrix_set(X, i, j, 0.0f);
            }
        }

    }
};


void matrix_sigmoid(Matrix* X){
    if (X->n_cols){
  
        for (size_t i = 0; i < X->n_rows; i++){
            for (size_t j = 0; j < X->n_cols; j++){
                double x = matrix_get(X, i, j);
                matrix_set(X, i, j, 1/(1 + exp(-x)));
            }
        }
    }
};

void matrix_tanh(Matrix* X){
    if (X->n_cols){
        for (size_t i = 0; i < X->n_rows; i++){
            for (size_t j = 0; j < X->n_cols; j++){
                double x = matrix_get(X, i, j);
                matrix_set(X, i, j, exp(2*x - 1)/exp(2*x + 1));
            }
        }
    }
};

void matrix_linear(Matrix* X){
};
#pragma endregion Activation Functions

#endif // ACT_FN_H_