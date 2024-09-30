#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include "math.h"
#include "matrix.h"
#include "layers.h"
#include "string.h"

#pragma region Optimizer

#pragma region Adam
/*
Adam Optimizer 

Adaptive Moment Estimation is an algorithm for optimization technique for gradient descent. 
The method is really efficient when working with large problem involving a lot of data or parameters.
It requires less memory and is efficient. Intuitively, it is a combination of the ‘gradient descent 
    with momentum’ algorithm and the ‘RMSP’ algorithm. 
A combination of two gradient descent methodoligies

Momentum: 

This algorithm is used to accelerate the gradient descent algorithm by taking into consideration 
the ‘exponentially weighted average’ of the gradients. Using averages makes the algorithm converge
towards the minima in a faster pace. ​
                                w_t+1 = w_t - alpha * m_t
                          where m_t = beta_1 * m_t-1 + (1-beta_1)*[delta_W[i][t]] (1)


Root Mean Square Propagation:

Root mean square prop or RMSprop is an adaptive learning algorithm that tries to improve AdaGrad.
Instead of taking the cumulative sum of squared gradients like in AdaGrad, 
    it takes the ‘exponential moving average’.
                                w_t+1 = w_t - (alpha_t)/(v_t + epsilon)^(0.5) * [delta_W[i][t]]
                          where v_t = beta_2*v_t-1 + (1-beta_2)*[delta_W[i][t]]^2 (2)

Since mt and vt have both initialized as 0 (based on the eq. (1) and (2)),
it is observed that they gain a tendency to be ‘biased towards 0’ as both β1 & β2 ≈ 1.
This Optimizer fixes this problem by computing ‘bias-corrected’ mt and vt.
This is also done to control the weights while reaching the global minimum to prevent 
    high oscillations when near it. 

The formulas used are:
                                m_dach_t = m_t/(1-beta_1^t)
                                v_dach_t = v_t/(1-beta_2^t)

        --> w_t+1 = w_t - m_dach_t*(alpha/sqrt(v_dach_t + epsilon))
*/

typedef struct{
    double learning_rate;
    double alpha;
    double beta_1;
    double beta_2;
    double epsilon;
    size_t num_layers;
    Matrix* m_w_ptr;
    Matrix* m_b_ptr;
    Matrix* v_w_ptr;  
    Matrix* v_b_ptr;
}Adam_Optimizer;

void init_Adam_optimizer(Adam_Optimizer** optimizer_dptr, const double lr, const double alpha, const double beta_1,  const double beta_2, const double epsilon, Layer* layers, const size_t num_layers){
    if (*optimizer_dptr == NULL){
        *optimizer_dptr = (Adam_Optimizer*)malloc(sizeof(Adam_Optimizer)); 
    }

    (*optimizer_dptr)->learning_rate = lr;
    (*optimizer_dptr)->alpha = alpha;
    (*optimizer_dptr)->beta_1 = beta_1;
    (*optimizer_dptr)->beta_2 = beta_2;
    (*optimizer_dptr)->epsilon = epsilon;
    (*optimizer_dptr)->num_layers = num_layers;

    // Allocate space for gradients for weights and biases in each layer
    (*optimizer_dptr)->m_w_ptr = (Matrix*)malloc(num_layers*sizeof(Matrix));
    (*optimizer_dptr)->m_b_ptr = (Matrix*)malloc(num_layers*sizeof(Matrix)); 
    (*optimizer_dptr)->v_w_ptr = (Matrix*)malloc(num_layers*sizeof(Matrix));
    (*optimizer_dptr)->v_b_ptr = (Matrix*)malloc(num_layers*sizeof(Matrix));
    
    // Initialize
    for (size_t i = 0; i < num_layers; i++){
        Layer* layer_ptr = layers + i;

        switch(layer_ptr->type){
            case FEED_FORWARD:
                FeedForwardLayer* ff_layer_ptr = layer_ptr->layer.ff_layer;
                
                //set rows and cols
                ((*optimizer_dptr)->m_w_ptr + i)->data = (double*)calloc(ff_layer_ptr->grad_W->n_rows * ff_layer_ptr->grad_W->n_cols, sizeof(double));
                ((*optimizer_dptr)->m_w_ptr + i)->n_rows = ff_layer_ptr->grad_W->n_rows; 
                ((*optimizer_dptr)->m_w_ptr + i)->n_cols = ff_layer_ptr->grad_W->n_cols;
                                
                ((*optimizer_dptr)->v_w_ptr + i)->data = (double*)calloc(ff_layer_ptr->grad_W->n_rows * ff_layer_ptr->grad_W->n_cols, sizeof(double));
                ((*optimizer_dptr)->v_w_ptr + i)->n_rows = ff_layer_ptr->grad_W->n_rows;
                ((*optimizer_dptr)->v_w_ptr + i)->n_cols = ff_layer_ptr->grad_W->n_cols;

                ((*optimizer_dptr)->m_b_ptr + i)->data = (double*)calloc(ff_layer_ptr->grad_b->n_rows * ff_layer_ptr->grad_b->n_cols, sizeof(double));
                ((*optimizer_dptr)->m_b_ptr + i)->n_rows = ff_layer_ptr->grad_b->n_rows;
                ((*optimizer_dptr)->m_b_ptr + i)->n_cols = ff_layer_ptr->grad_b->n_cols;   

                ((*optimizer_dptr)->v_b_ptr + i)->data = (double*)calloc(ff_layer_ptr->grad_b->n_rows * ff_layer_ptr->grad_b->n_cols, sizeof(double));
                ((*optimizer_dptr)->v_b_ptr + i)->n_rows = ff_layer_ptr->grad_b->n_rows;
                ((*optimizer_dptr)->v_b_ptr + i)->n_cols = ff_layer_ptr->grad_b->n_cols;

                break;

            default:
                printf("Provided layer type not supported.\n");
                exit(0);     

        }   
    }

};

void optimize_adam(Adam_Optimizer* optimizer, Layer* layers){
    
    double m_t_prev, m_t, m_dach_t, v_t, v_t_prev;
    double grad_W_j_k_t, w_j_k_opt, v_dach_t, w_j_k_t;
    double grad_b_j_t, b_j_opt, b_j_t;

    for (size_t i = 0; i < optimizer->num_layers; i++){
        Layer* layer_ptr = layers + i;
        switch(layer_ptr->type){
            case 0:
                FeedForwardLayer* ff_layer_ptr = layer_ptr->layer.ff_layer;
                for (size_t j = 0; j < ff_layer_ptr->grad_W->n_rows; j++){
                    for (size_t k =0; k < ff_layer_ptr->grad_W->n_cols; k++){
                        // Weights                       
                            grad_W_j_k_t = matrix_get(ff_layer_ptr->grad_W, j, k);
                            
                            // m_W
                                m_t_prev = matrix_get(optimizer->m_w_ptr + i, j, k);
                            
                            
                                // Calculate value of m_t+1 and update m_t
                                m_t = optimizer->beta_1 * m_t_prev + (1 - optimizer->beta_1) * grad_W_j_k_t; // delta_W[j][k]
                                matrix_set(optimizer->m_w_ptr + i, j, k, m_t); 
                                
                                m_dach_t = m_t/(1 - pow(optimizer->beta_1, i));
    
                            // v_W
                                v_t_prev = matrix_get(optimizer->v_w_ptr, j, k);
                                
                                // Calculate value of v_t+1 and update v_t
                                v_t = optimizer->beta_2 * v_t_prev + (1- optimizer->beta_2) * pow(grad_W_j_k_t, 2); // [delta_W[i][t]]^2
                                matrix_set(optimizer->v_w_ptr + i, j, k, v_t);

                                v_dach_t = v_t/(1 - pow(optimizer->beta_2, i));

                        // update the corresponding weight W[j][k] of the Layer i of the model
                            w_j_k_t = matrix_get(ff_layer_ptr->weights, j, k);
                            w_j_k_opt = w_j_k_t - m_dach_t * (optimizer->alpha / sqrt(v_dach_t + optimizer->epsilon));
                            matrix_set(ff_layer_ptr->weights, j, k, w_j_k_opt);

                    }
                        // Biases                        
                            grad_b_j_t = matrix_get(ff_layer_ptr->grad_b, j, 0);
                            
                            // m_b
                                m_t_prev = matrix_get(optimizer->m_b_ptr + i, j, 0);
                            
                            
                                // Calculate value of m_t+1 and update m_t
                                m_t = optimizer->beta_1 * m_t_prev + (1 - optimizer->beta_1) * grad_b_j_t; // delta_W[j][k]
                                matrix_set(optimizer->m_b_ptr + i, j, 0, m_t); 
                                
                                m_dach_t = m_t/(1 - pow(optimizer->beta_1, i));
    
                            // v_b
                                v_t_prev = matrix_get(optimizer->v_b_ptr, j, 0);
                                
                                // Calculate value of v_t+1 and update v_t
                                v_t = optimizer->beta_2 * v_t_prev + (1 - optimizer->beta_2) * pow(grad_b_j_t, 2); // [delta_W[i][t]]^2
                                matrix_set(optimizer->v_b_ptr + i, j, 0, v_t);

                                v_dach_t = v_t/(1 - pow(optimizer->beta_2, i));

                        // update the corresponding weight b[j] of the Layer i of the model
                            b_j_t = matrix_get(ff_layer_ptr->biases, j, 0);
                            b_j_opt = b_j_t - m_dach_t * (optimizer->alpha / sqrt(v_dach_t + optimizer->epsilon));
                            matrix_set(ff_layer_ptr->biases, j, 0, b_j_opt);
                }
        }
    }   
}

void destroy_adam_optimizer(Adam_Optimizer* optimizer){
    if (optimizer == NULL) return;
    for (int i = optimizer->num_layers - 1; i >= 0; i--){

        // Free matrices
            matrix_destroy(optimizer->m_w_ptr + i);
            matrix_destroy(optimizer->v_w_ptr + i);
            matrix_destroy(optimizer->m_b_ptr + i);
            matrix_destroy(optimizer->v_b_ptr + i);
    }

    // Free addresses
        free(optimizer->m_w_ptr);
        free(optimizer->v_w_ptr);
        free(optimizer->m_b_ptr);
        free(optimizer->v_b_ptr);
};

#pragma endregion Adam

#pragma endregion Optimizer

#endif // OPTIMIZER_H_