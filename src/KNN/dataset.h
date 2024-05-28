#ifndef DATASET_H_
# define DATASET_H_


#include "vector.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"

typedef struct{
	Vector* vec;
	unsigned char num_classes;
	
}Dataset;

Dataset* dataset_create(){
	Dataset* dataset = (Dataset*) malloc(sizeof(Dataset));
	dataset->vec = NULL;
	dataset->num_classes = 0;
	return dataset;	
};

void dataset_set_vector(Dataset* dataset, Vector* vec){
	dataset->vec = vec;
};

void dataset_set_num_classes(Dataset* dataset, unsigned char num_classes){
	dataset->num_classes = num_classes;
};

void dataset_initialize(Dataset* dataset, unsigned short int initial_capacity){
	dataset->vec = vector_create(initial_capacity);
	//memcpy(dataset->vec, vector_create(initial_capacity), sizeof(Vector));
};

void dataset_populate(Dataset* dataset){
	printf("Populating dataset\n");
	unsigned short int N;
	float max_x, max_y;
	
	printf("Enter the number of Points:\n");
	scanf("%hu", &N);

	printf("Enter Upper magnitude limit for x:\n");
	scanf("%f", &max_x);

	printf("Enter Upper magnitude limit for y:\n");
	scanf("%f", &max_y);
	
	// Create vector
	Point* points[N];

	// seed the random number generator
	srand(time(NULL));
	float random_x, random_y;
	unsigned char r = 255, g = 255, b = 255;

	for (unsigned short int i = 0; i < N; i++){
		random_x = 2 * (((float)rand() / RAND_MAX) - 0.5); // map between -1 and 1
		random_x *= max_x;
		
		random_y = 2 * (((float)rand() / RAND_MAX) - 0.5); // map between -1 and 1
		random_y *= max_y;
		
		// Create point
		Point* p = point_create();
		point_set_point(p, &random_x, &random_y);

		// Set color
		Color* color = color_create();
		color_set_rgb(color, &r, &g, &b);
		point_set_color(p, color);		

		// push the point
		//vector_push_back(v, p);
		points[i] = p;
	}
	// Initialize the dataset
	dataset_initialize(dataset, N);

	for (unsigned short int i = 0; i < N; i++){
		vector_push_back(dataset->vec, points[i]);
		point_destroy(points + i);
	}

	free(*points);
	*points = NULL;
	printf("Dataset populated\n");
};		

void dataset_read_csv(Dataset* dataset, const char* filename, unsigned short int N){
	FILE* file = fopen(filename, "r");
	if (file == NULL){
		printf("Error opening file\n");
		return;
	}
		
	
	unsigned char class;
	Point* points = (Point*) malloc(sizeof(Point));
	float vals[4];
	unsigned char num_classes = 0;
	
	char buffer[1024];
	size_t i = 0;
	size_t j = 0;
	size_t num_features = 0;

	printf("Reading file\n");
	while (fgets(buffer, 1024, file)){
		//printf("Reading line %lu\n", i);
		char *token = strtok(buffer, ",");
		if (i){
			j %= num_features;
		}

		//printf("i: %lu\n", i);
		
		while(token){
			//printf("j: %lu\n",j);
			if (!i){
				num_features++;
			}
			else{

				// modify this part for the data you have
				switch(j){
					case 4:
						class = (unsigned char)atoi(token);
						break;
					default:
						vals[j] = atof(token);
						break;
				}

			}
			j++;
			
			token = strtok(NULL, ",");
		}

		points = realloc(points, (i + 1) * sizeof(Point));
		Point* p = point_create();

		// Read only the first two features -> 2D
		point_set_point(p, vals, vals + 1);
		point_set_class(p, class);
		points[i] = *p; 

		// Count the number of classes
		if (class > num_classes){
			num_classes = class;
		}

		free(p);

		i++;

	}	
	
	// Set the actual number of classes
	dataset->num_classes = ++num_classes;

	dataset_initialize(dataset, N);
	
	// Push the points to the dataset
	for (i = 0; i < N; i++){
		vector_push_back(dataset->vec, points + i);
	}
	
	free(points);
	points = NULL;
	fclose(file);
};

void dataset_destroy(Dataset* dataset){
	//printf("Destroying dataset\n");
	vector_destroy(dataset->vec);
	dataset->vec = NULL;
	free(dataset);
	dataset = NULL;

};

void dataset_print(Dataset* dataset){
	vector_print(dataset->vec);
};

void dataset_split(Dataset* dataset, Dataset* train, Dataset* test, float ratio){
	unsigned short int N = dataset->vec->size;
	unsigned short int train_size = (unsigned short int)(ratio * N);
	unsigned short int test_size = N - train_size;
	
	// Initialize the train and test datasets
	train->vec = vector_create(train_size);
	test->vec = vector_create(test_size);
	
	// Shuffle the dataset
	vector_shuffle(dataset->vec);
	
	// Copy the points to the train and test datasets
	for (unsigned short int i = 0; i < train_size; i++){
		vector_push_back(train->vec, vector_at(dataset->vec, i));
	}
	
	for (unsigned short int i = train_size; i < N; i++){
		vector_push_back(test->vec, vector_at(dataset->vec, i));
	}
	
	train->num_classes = dataset->num_classes;
	test->num_classes = dataset->num_classes;
};
#endif
