## Introduction
This repository performs individual detection in consideration of overlap by using CNN in pixel units.


## Network
This model consists of three convolution layers (conv1-conv3), two max-pooling layers (pool1 and pool2), and four fully connected layers (fc1-fc4). Conv1 has 7×7×3 filters, conv2 has 7×7×32 filters and conv3 has 5×5× 32 filters. Max pooling layers with 2×2 kernel size are used after conv1 and conv2. Batch normalization was applied to conv1-con3 and fc1-fc3. The activate function was applied after every convolutional layer and fully connected layer by Leaky ReLU.
<img src="./image/demo/model.png" alt="model">

## Getting Started
### Install Required Packages
First ensure that you have installed the required packages (requirements.txt).  


## Training
To create learning data in the following procedure.
1. To get images from movie. To input the path of movie file and the path of output image file.
```
make
```
or
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

->