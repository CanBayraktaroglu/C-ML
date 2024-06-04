#include "dataset.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "dt.h"
#include "utils.h"

void main(void){

	//Instantiate a decision tree
	DT_Node* root = dt_node_create();
	//Instantiate a dataset
	Dataset* dataset = dataset_create();

	// Configs
	DT_Config config;
	load_yaml_dt("../src/DT/configs/configs.yaml", &config);// load

	//Initialize the dataset
	dataset_read_csv(dataset, config.data_path);

	//build decision tree
	dt_node_build(root, dataset->vec, dataset->num_classes);

	//Free memory
	dataset_destroy(dataset);
	dataset = NULL;
	dt_node_destroy(root, 0);
	root = NULL;

}	