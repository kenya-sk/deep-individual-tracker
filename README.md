# CNN

### src
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

5. cnn.py
    * 6 layer CNN. this model learn the equation that converts **each local image** of
    input image into a density map.

6. cnn_pixel.py
    * 7 layer CNN. this model learns an equation that converts **each pixel** of input
    image into a density map.

7. estimate_pixel.py
    * using the learned model(cnn_pixel.py), estimate the value of
    the density map for each pixel. after that, clustering by MeanShift.

8. accuracy.py
    * calculate the accuracy. if the distance between the estimation and ground truth is
    less than threshold(arg:distThreshold), it is regarded as the correct estimation.
