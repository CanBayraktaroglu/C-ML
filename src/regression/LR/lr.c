#include "utils.h"
#include "matrix.h"

void main(void){
    Matrix* X = matrix_create(3, 2);
    matrix_set(X, 0, 0, 1);
    matrix_set(X, 0, 1, 2);
    matrix_set(X, 1, 0, 3);
    matrix_set(X, 1, 1, 4);
    matrix_set(X, 2, 0, 5);
    matrix_set(X, 2, 1, 6);
    matrix_print(X);
    printf("----------------\n");
    Matrix* X_T = matrix_transpose(X);
    matrix_print(X_T);
    matrix_destroy(&X);
    matrix_destroy(&X_T);
};