# Large-Scale-Palm-Tree-Detection-in-High-Resolution-Satellite-Images-Using-U-Net
Python implementation of Convolutional Neural Network (CNN) proposed in academic paper

This repository includes functions to preprocess the input images and their respective polygons so as to create the input image patches 
and mask patches to be used for model training. The CNN used here is the modified u - net model implemented in the paper 
'Large Scale Palm Tree Detection in High Resolution Satellite Images Using U - Net' by Freudenberg M., Nolke N., Agostini A., Urban K., 
Worgotter F., Kleinn C. (2019).

The main differences between the implementations in the paper and the implementation in this repository is as follows:

- Sigmoid layer is used as the last layer instead of the softmax layer, in consideration of the fact that this is a binary classification 
  problem  
- The dice coefficient function is used as the loss function in place of the binary cross - entropy loss function, in consideration of the
  fact that this is a semantic segmentation problem, whereby emphasis should be placed on accuracy of target delineation  
- No cropping is done for both training and inference processes, in order to speed up the prediction process without significant 
  accuracy loss

Requirements:
- cv2
- glob
- json
- numpy
- rasterio
- keras (tensorflow backend)
