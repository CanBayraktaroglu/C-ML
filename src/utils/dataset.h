#ifndef DATASET_H_
# define DATASET_H_


#include "vector.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"
#include "matrix.h"
#include "time.h"

typedef struct{
	Vector* vec;
	unsigned char num_classes;
	
}Dataset;

typedef struct{
	Matrix* data;
	size_t N;
}Dataset_;

Dataset_* Dataset_New(){
	Dataset_* dataset = (Dataset_*)malloc(sizeof(Dataset_));
	return dataset;
}; 

void dataset_initialize_(Dataset_* dataset, size_t N){
	dataset->N = N;	
	dataset->data = (Matrix*)malloc(N * sizeof(Matrix));
};

Dataset* dataset_create(){
	Dataset* dataset = (Dataset*) malloc(sizeof(Dataset));
	dataset->vec = NULL;
	dataset->num_classes = 0;
	return dataset;	
};

void dataset_set_vector(Dataset* dataset, Vector* vec){
	if (dataset->vec == NULL){
		dataset->vec = (Vector*) malloc(sizeof(Vector));
	}
	memcpy(dataset->vec, vec, sizeof(Vector));
	//dataset->vec = vec;
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
		Point* p = point_create(2);
		float point[2] = {random_x, random_y};
		point_set_point(p, point);

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

void dataset_read_csv(Dataset* dataset, const char* filename){
	FILE* file = fopen(filename, "r");
	if (file == NULL){
		printf("Error opening file\n");
		return;
	}

	unsigned char class;
	unsigned char num_classes = 0;
	char buffer[1024];
	size_t i = 0;
	size_t j = 0;
	size_t num_features = 0;

	// Traverse the file
	while (fgets(buffer, 1024, file)){
		char *token = strtok(buffer, ",");
		while(token){
			if (!i){
				num_features++;
			}

			token = strtok(NULL, ",");
		}

		i++;
	}	

	float* vals = (float*) malloc(sizeof(float) * (num_features - 1));
	rewind(file);
	unsigned short int N = (unsigned short int)(i-1);
	printf("Number of points: %hu\n", N);
	Point* points[N];
	i = 0;

	printf("Reading file\n");
	while (fgets(buffer, 1024, file)){
		char* token = strtok(buffer, ",");
		if (!i){
			i++;
			continue;
		}

		j %= num_features;
		while(token){
			
			// modify this part for the data you have
			switch(j){
				case 4:
					class = (unsigned char)atoi(token);
					break;
				default:
					vals[j] = atof(token);
					break;
			}			
		
			j++;
			token = strtok(NULL, ",");
		}

		Point* p = point_create((unsigned char)(num_features - 1));

		point_set_point(p, vals);
		point_set_class(p, class);
		points[i - 1] = p;
		
		// Count the number of classes
		if (i && class > num_classes){
			num_classes = class;
		}
			
		i++;

	}	
	
	// Set the actual number of classes
	dataset->num_classes = ++num_classes;

	dataset_initialize(dataset, N);
	
	// Push the points to the dataset
	for (i = 0; i < N; i++){
		vector_push_back(dataset->vec, points[i]);
		free(points[i]);
		
	}
	*points = NULL;
	free(vals);
	vals = NULL;
	fclose(file);
};

void dataset_destroy(Dataset** dataset){
	if (*dataset == NULL) return;

	vector_destroy(&(*dataset)->vec);
	(*dataset)->vec = NULL;

	free(*dataset);
	*dataset = NULL;

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
	
	for (unsigned short int j = train_size; j < N; j++){
		vector_push_back(test->vec, vector_at(dataset->vec, j));
	}
	
	train->num_classes = dataset->num_classes;
	test->num_classes = dataset->num_classes;
};

void dataset_shuffle_(Dataset_* dataset){
	Matrix* tmp_1 = (Matrix*)malloc(sizeof(Matrix));
	Matrix* tmp_2 = (Matrix*)malloc(sizeof(Matrix));

	for (size_t i = 0; i < dataset->N; i++){
		size_t j = rand() % dataset->N;

		// copy matrix at i to temporary buffer 1
		memcpy(tmp_1, dataset->data + i, sizeof(Matrix));
		matrix_destroy(dataset->data + i);

		// copy matrix at j to temporary buffer 2
		memcpy(tmp_2, dataset->data + j, sizeof(Matrix));
		matrix_destroy(dataset->data + j);

		//copy matrix at tmp_1 to memory at j
		memcpy(dataset->data + j, tmp_1, sizeof(Matrix));

		//copy matrix at tmp_2 to memory at i
		memcpy(dataset->data + i, tmp_2, sizeof(Matrix));

		// erase memory in the temporary buffers
		matrix_destroy(tmp_1);
		matrix_destroy(tmp_2);

	}	

	free(tmp_1);
	tmp_1 = NULL;
	free(tmp_2);
	tmp_2 = NULL;
};

void dataset_split_(Dataset_* dataset, Dataset_* train, Dataset_* test, double ratio){
	size_t N = dataset->N;
	size_t train_size = (size_t)(ratio*N);
	size_t test_size = N - train_size;

	// shuffle dataset
	dataset_shuffle_(dataset);

	// Copy matrices to the train dataset
	Matrix* matrix_ptr = NULL;
	for (size_t i = 0; i < N; i++){
		matrix_ptr = dataset->data + i;

		if (i < train_size){
			memcpy(train->data + i, matrix_ptr, sizeof(Matrix));
			continue;
		}

		memcpy(test->data + i, matrix_ptr, sizeof(Matrix));
	}
};
#endif
