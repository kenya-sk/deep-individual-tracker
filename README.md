## Introduction
This repository performs individual detection in consideration of overlap by using CNN in each pixel.


## Network
This model consists of three convolution layers (conv1-conv3), two max-pooling layers (pool1 and pool2), and four fully connected layers (fc1-fc4). Conv1 has 7×7×3 filters, conv2 has 7×7×32 filters and conv3 has 5×5× 32 filters. Max pooling layers with 2×2 kernel size are used after conv1 and conv2. Batch normalization was applied to conv1-con3 and fc1-fc3. The activate function was applied after every convolutional layer and fully connected layer by Leaky ReLU.
<img src="./image/demo/model.png" alt="model" height= 400 vspace="25" hspace="70">

## Getting Started
### Install Required Packages
First ensure that you have installed the required packages (requirements.txt).  

## Create Datasets
Files required for creating datasets are under "./src/datasets".

```
1. my_make.sh
    To remove the file of the previous version, 
    and create a new file according to Makefile.

2. video2img.cpp
    Extract images from the video at a regular interval and save.

3. img2train.py
    Load the image and click on the target to make an answer label (density map).

4. video2train.py
    Load the video and click on the target to make an answer label (density map).  
    The output results are the same, but 1-3 processing can be done directly the video.

5. cord2dens.py
    It receives coordinates and plots a density map with arbitrary kernel width.  
    (For testing when finding an appropriate kernel width.)

6. pred2video.py
    To convert sequential images to video.
   (For checking if individual detection is successful with the trained model.) 
```

## Training
To create training data in the following procedure. Each file is included in the directory (src/datasets).
1. To get images from movie. To input the path of movie file and the path of output image file.
```
my_make.sh
```
or
```
g++ -o video2image video2image.cpp -std=c++11 `pkg-config --cflags opencv` `pkg-config --libs opencv`
```

2. By clicking the image, labeling is done and training data is created. To input the path of image file. To press "D" when you mistake annotation
```
python3 img2train.py
```

3. If you want training data of arbitrary frames, to create it by the following procedure.  
To input the path of video file. To press "P" when the desired frame appear. To press "D" when you mistake annotation. To press "S" when you finish selecting feature point and want to save. To press "Q" when you want to finish creating training data.
```
python3 video2train.py
```


## Training Your Own Model
To train your own models, follow these steps. Each file is included in the directory (src/model).  
The following technique was used to train the model.
- Down Sampling
- Early Stopping
- Batch Normalization
- Data Augmentation (Horizontal flip)
- Hard Negative Mining

```
model.py
    The Architecture of the model is defined.

train.py
    Training the model.

cnn_util.py
    The util function.

clustering.py
    To clustering the density maps.

predict.py
    Individual detection is performed using a learned model. 
    
evaluation.py
    Evaluate with Accuracy, recall, Precision, and F-measure.
```

### Training
To set the path of training/validation image and path of answer data(density map).  
Each parameter can be set with argument.
```
python3 train.py [-h]  [--root_img_dirc]  [--root_dens_dirc]  [--mask_path]
                 [reuse_model_path]  [--root_log_dirc]  [save_model_dirc]
                 [--visible_device]  [--memory_rate]  [--test_size]  [--local_img_size]
                 [--n_epochs]  [--batch_size]  [--min_epoch]  [--stop_count]
                 [flip_prob]  [--under_sampling_thresh]
```

Hyper parameters and learning conditions can be set using arguments. Details are as follows.

```
[-h]                      : help option
[--root_img_dirc]         : The path of training dataset
[--root_dens_dirc]        : The path of answer label (density map)
[--mask_path]             : The path of mask image
[reuse_model_path]        : The path of pretrained model
[--root_log_dirc]         : The path of tensor board log
[save_model_dirc]         : The path of save the weight of trained model
[--visible_device]        : The ID of using GPU
[--memory_rate]           : The ratio of using GPU memory
[--test_size]             : The ratio of test data
[--local_img_size]        : The size of local image (shape is square)
[--n_epochs]              : The maximum number of epochs
[--batch_size]            : Batch size
[--min_epoch]             : The minimum number of epochs
[--stop_count]            : If not_improved_count < stop_count, execute early stopping
[--flip_prob]             : The ratio of horizontal flip flop 
[--under_sampling_thresh] : The threshold of positive data
```

### Prediction
To perform individual detection with the trained model.  
Each parameter can be set with argument.
```
python3 predict.py
```


### Evaluation
The evaluation metrics of individual detection are accuracy, recall, precision, and F-measure. The position of the individual predicted by CNN and the position of the answer label were associated by using the Hungarian algorithm. True Positive defined ad If the matching distance was less than or equal to the threshold value (default is 15).

```
python3 evaluation.py
```
