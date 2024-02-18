# Oil_spill_identification_from_SAR_images

This repository contains an implementation of the CNN described in the paper [*Oil Spill Identification from SAR Images for Low Power Embedded Systems Using CNN*](https://www.mdpi.com/2072-4292/13/18/3606).\
The paper describes a CNN to perform semantic segmentation of SAR images to enable early oil spill detection while maintaining a low number of parameters and a low-inference time.
The CNN is able to identify five different classes: Sea, Oil Spill, Look-Alike, Ship, and Land.

The following image (taken from the original paper) shows an example of CNN's input, ground truth, and output.
![Example of input, output, and ground truth of the CNN. Image taken from the original paper](https://www.mdpi.com/remotesensing/remotesensing-13-03606/article_deploy/html/images/remotesensing-13-03606-g004.png)
