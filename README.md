# Oil_spill_identification_from_SAR_images

This repository contains an implementation of the CNN described in the paper [*Oil Spill Identification from SAR Images for Low Power Embedded Systems Using CNN*](https://www.mdpi.com/2072-4292/13/18/3606).\
The paper describes a CNN to perform semantic segmentation of SAR images to enable early oil spill detection while maintaining a low number of parameters and a low-inference time.
The CNN is able to identify five different classes: Sea, Oil Spill, Look-Alike, Ship, and Land.

To train the model copy the original dataset images and labels as follow:
(to see the dataset used, see the paper)
```
dataset
├── original_data
    ├── test
    │   ├── images
    │   │   ├── sample_000.jpg
    │   │   ├── ...
    │   └── labels
    │       ├── sample_000.png
    │       ├── ...
    └── train
        ├── images
        │   ├── sample_000.jpg
        │   ├── ...
        └── labels
            ├── sample_000.png
            ├── ...
```

Then run:
```
python3 generate_oil_dataset.py
./generate_file_list.sh
python3 train_model.py
```

Use test_model.py to test the trained model.
Use the help option to learn the relevant parameters:
```
python3 test_model.py -h
```

The following image (taken from the original paper) shows an example of CNN's input, ground truth, and output.
![Example of input, output, and ground truth of the CNN. Image taken from the original paper](https://www.mdpi.com/remotesensing/remotesensing-13-03606/article_deploy/html/images/remotesensing-13-03606-g004.png)
