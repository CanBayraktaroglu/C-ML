#ifndef COLOR_H_
# define COLOR_H_

#include <stdlib.h>
#include <stdio.h>

typedef struct{
	unsigned char r;
	unsigned char g;
	unsigned char b;
}Color;

Color* color_create(){
	Color* color = (Color*) malloc(sizeof(Color));
	return color;
};

void color_set_rgb(Color* color, unsigned char* r, unsigned char* g, unsigned char* b){
	color->r = *r;
	color->g = *g;
	color->b = *b;
};

void color_destroy(Color* color){
	//printf("Destroying color\n");
	if (color == NULL ) return;
	free(color);
	//printf("Color destroyed\n");
	color = NULL;
};

#endif