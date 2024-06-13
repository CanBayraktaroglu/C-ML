#ifndef __SVM_H__
#define __SVM_H__

#include <stdlib.h>
#include <string.h>	// memcpy

typedef struct {
    float *w;
    float b;
    unsigned char dim;
} LinearSVM;

LinearSVM* svm_create(unsigned char dim){
    LinearSVM *svm = (LinearSVM*)malloc(sizeof(LinearSVM));
    svm->w = NULL;
    svm->b = 0.0f;
    svm->dim = dim;
    return svm;
}

void set_w(LinearSVM *svm, float *w){
    if (svm->w == NULL){
        svm->w = (float*)malloc(sizeof(float) * svm->dim);
    }

    memcpy(svm->w, w, sizeof(float) * svm->dim);
}


#endif