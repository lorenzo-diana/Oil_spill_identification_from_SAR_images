#!/bin/bash
declare -r training_set_size=70
declare -r validation_set_size=30
declare -r DATASET_HOME_FOLDER='./dataset'
ls $DATASET_HOME_FOLDER'/train_tile_aug/' | shuf > ./temp_list
num_samples=$(ls $DATASET_HOME_FOLDER'/train_tile_aug/' | wc -l)
head -n $(($num_samples*$training_set_size/100)) ./temp_list > ./train_set.txt
tail -n $(($num_samples*$validation_set_size/100)) ./temp_list > ./validation_set.txt
rm ./temp_list
#ls $DATASET_HOME_FOLDER'/test_tile_aug/' > ./test_set.txt
ls $DATASET_HOME_FOLDER'/test_tile/' > ./test_set.txt
