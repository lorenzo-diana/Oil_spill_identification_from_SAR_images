#!/bin/bash
export training_set_size=70
export validation_set_size=30
ls ./dataset/train_tile_aug/ | shuf > ./temp_list
num_samples=$(ls ./dataset/train_tile_aug/ | wc -l)
head -n $(($num_samples*$training_set_size/100)) ./temp_list > ./train_set.txt
tail -n $(($num_samples*$validation_set_size/100)) ./temp_list > ./validation_set.txt
rm ./temp_list
ls ./dataset/test_tile_aug/ > ./test_set.txt
