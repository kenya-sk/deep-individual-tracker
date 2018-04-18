## Introduction

## Source code
1. movie2image.cpp
    * get images from movie at arbitrary interval (default: 1 min).

2. movie2trainingData.py
    * load movie, select feature points with a mouse for specific frames,
    and output the coordinate of density map (.npy).

3. image2trainingData.py
    * input the directory path, select feature points with a mouse for all image file,
    and output the coordinate of density map (.npy).

4. cord2dens.py
    * input the directory path of cordinate file(.csv), output density map of various kernel size (.npy).

5. cnn_pixel.py
    * 7 layer CNN. this model learns an equation that converts **each pixel** of input
    image into a density map. and, estimate using learned model.

6. clustering.py
    * clustering by MeanShift. the centroid of cluster is the detection point.

7. accuracy.py
    * calculate the accuracy. if the distance between the estimation and ground truth is
    less than threshold(arg:distThreshold), it is regarded as the correct estimation.


## Requirement
    * Python3 or more
    * C++
    * OpenCV
    * TensorFlow

## Training Your Own Model
