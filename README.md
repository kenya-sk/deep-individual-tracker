## Introduction

## Requirement
* Python3
* C++
* OpenCV
* TensorFlow

## Source code
1. movie2image.cpp
    * To get images from movie at arbitrary interval (default: 1 min).

2. movie2trainingData.py
    * First, load movie. Second, Select feature points with a mouse for specific frames,
    and output the coordinate of density map (.npy).

3. image2trainingData.py
    * Input the directory path, select feature points with a mouse for all image file,
    and output the coordinate of density map (.npy).

4. cord2dens.py
    * Input the directory path of cordinate file (.csv), output density map of various kernel size (.npy).

5. cnn_pixel.py
    * This model (7 layer CNN) learns an equation that converts **each pixel** of input
    image into a density map. And, estimate using learned model.

6. clustering.py
    * Clustering by MeanShift. the centroid of cluster is the detection point.

7. accuracy.py
    * To calculate the accuracy. If the distance between the estimation and ground truth is
    less than threshold (arg:distThreshold), it is regarded as the correct estimation.


## Learning Data
To create learning data in the following procedure.
1. To get images from movie. To input the path of movie file and the path of output image file.
```
g++ -o movie2image.cpp `pkg-config --cflags opencv` `pkg-config --libs opencv`
```

2. By clicking the image, labeling is done and learning data is created. To input the path of image file.
```
python3 image2trainingData.py
```

3. If you want learning data of arbitrary frames, to create it by the following procedure. To input the path of movie file. To press "P" when the desired frame appear.
```
python3 movie2trainingData.py
```

## Training Your Own Model
To train your own models, follow these steps.  

### Training
To set the path of learning image and path of answer data(density map).
```
python3 cnn_pixel.py
```

### Estimation
To set path of learned model. And, set estimation value to True. It is argument of main function.
```
python3 cnn_pixel.py
```

### Clustering
To set path of estimation file (numpy file) and path of image.
```
python3 clustering.py
```

### Evaluation
To set path of estimation file and path of groundTruth file.
```
python3 accuracy.py
```
