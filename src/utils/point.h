#ifndef POINT_H_
# define POINT_H_

#include "color.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

// define Functions

typedef struct{
	unsigned char dim;
	float* point;
	Color* color;
	unsigned char class;
}Point;


float* point_get_point(Point* p){
	return p->point;
};

void point_set_dim(Point* p, unsigned char dim){
	p->dim = dim;
};

void point_set_point(Point* p, float* point){
	if (p->point == NULL){
		p->point = (float*) malloc(sizeof(float) * (unsigned long)p->dim);		
	}
	
	memcpy(p->point, point, sizeof(float) * p->dim);
};

void point_set_color(Point* p, Color* color){
	//memcpy(p->color, color, sizeof(Color));
	p->color = color;
};

void point_set_class(Point* p, unsigned char class){
	p->class = class;
};

unsigned char point_get_K(Point* p){
	return p->class;
};

Color* point_get_color(Point* p){

	return p->color;
};

Point* point_create(unsigned char dim){
	Point* p = (Point*) malloc(sizeof(Point));
	p->class = 0;
	p->dim = dim;
	p->point = NULL;
	p->color = NULL;
	return p;

};

void point_set_feature_at(Point* p, unsigned char index, float value){
	if (index >= p->dim){
		printf("Error: Index out of bounds\n");
		exit(1);
	}
	p->point[index] = value;
};

float point_calc_dist(Point* p, Point* query){

	if (p->dim != query->dim){
		printf("Error: Dimensions do not match\n");
		printf("p->dim: %hhu\n", p->dim);
		printf("query->dim: %hhu\n", query->dim);
		exit(1);
	}

	float total = 0.0f;

	for (unsigned char i = 0; i < p->dim; i++){
		total += powf(p->point[i] - query->point[i], 2.0f);
	}
	
	total = sqrtf(total);

	return total;
};

void point_destroy(Point** p){
	if (*p == NULL) return;
	color_destroy((*p)->color);
	free((*p)->point);
	(*p)->point = NULL;
	free(*p);	
	*p = NULL;
};

void vec_point_destroy(Point** p){
	if (*p == NULL) return;
	color_destroy((*p)->color);
	free((*p)->point);
	(*p)->point = NULL;
};


#endif
