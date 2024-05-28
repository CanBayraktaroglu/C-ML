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
	float point[2];
	Color* color;
	unsigned char class;
}Point;

void point_set_point(Point* p, float* x, float* y){

	p->point[0] = *x;
	p->point[1] = *y;
};

float* point_get_point(Point* p){
	return p->point;
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

Point* point_create(){
	Point* p = (Point*) malloc(sizeof(Point));
	p->class = 0;
	p->color = NULL;
	return p;

};

float point_get_x(Point* p){
	return p->point[0];
};

float point_get_y(Point* p){
	return p->point[1];
};

float point_calc_dist(Point* p, Point* query){
	float x_p = point_get_x(p);
	float x_q = point_get_x(query);
	float y_p = point_get_y(p);
	float y_q = point_get_y(query); 
	
	float dist = powf(x_p - x_q, 2.0f) + powf(y_p - y_q, 2.0f);
	dist = sqrtf(dist);

	return dist;
};

void point_destroy(Point** p){
	if (*p == NULL) return;
	//printf("Destroying point\n");
	color_destroy((*p)->color);
	free(*p);	
	//printf("Point destroyed\n");
	*p = NULL;
};

#endif
