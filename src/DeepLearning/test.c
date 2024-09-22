#include "utils.h"
#include "matrix.h"

void main(void){
    Matrix* X = matrix_create(3, 3);
    matrix_set(X, 0, 0, 1);
    matrix_set(X, 0, 1, 2);
    matrix_set(X, 0, 2, 3);

    matrix_set(X, 1, 0, 4);
    matrix_set(X, 1, 1, 5);
    matrix_set(X, 1, 2, 6);

    matrix_set(X, 2, 0, 7);
    matrix_set(X, 2, 1, 8);
    matrix_set(X, 2, 2, 9);
    matrix_print(X);
    printf("----------------\n");

    matrix_destroy(&X);
};
