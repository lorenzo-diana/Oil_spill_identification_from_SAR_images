# Oil spill identification from SAR images

This repository contains an implementation of the CNN described in the paper [*Oil Spill Identification from SAR Images for Low Power Embedded Systems Using CNN*](https://www.mdpi.com/2072-4292/13/18/3606).\
The paper describes a CNN to perform semantic segmentation of SAR images to enable early oil spill detection while maintaining a low number of parameters and a low-inference time.
The CNN is able to identify five different classes: Sea, Oil Spill, Look-Alike, Ship, and Land.

The used dataset is described in subsection 4.1 of the above mentioned paper.
The dataset can be requested at the following link: [*Oil Spill Detection Dataset*](https://m4d.iti.gr/oil-spill-detection-dataset/)

# Requirements
This code has been validated with python 3.12 and the following packages are required:
```
keras==3.9.2
numpy==2.2.5
tensorflow==2.19.0
matplotlib==3.10.1
PyQt6==6.9.0
```

The full list of required packages can be found in the file: requirements.txt
To install the required packages:
```
pip install -r requirements.txt
```

# Usage
To train the model setup the following folders tree and place the original dataset as follow:

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

# Output example
The following image (taken from the original paper) shows an example of CNN's input, ground truth, and output.
![Example of input, output, and ground truth of the CNN. Image taken from the original paper](https://www.mdpi.com/remotesensing/remotesensing-13-03606/article_deploy/html/images/remotesensing-13-03606-g004.png)
