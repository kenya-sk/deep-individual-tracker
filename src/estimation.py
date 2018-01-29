#! /usr/bin/env python
# coding: utf-8

import math
import numpy as np
import tensorflow as tf
import cv2

import cnn

def get_local_image(image, localImgSize):
    # trimming original image(there are many unnecessary parts)
    image = image[:470, :]
    # local image size is even number
    height = image.shape[0]
    width = image.shape[1]
    pad = math.floor(localImgSize/2)
    if len(image.shape) == 3:
        padImg = np.zeros((height + pad * 2, width + pad * 2, image.shape[2]))
        localImg = np.zeros((localImgSize, localImgSize, image.shape[2]))
    else:
        padImg = np.zeros((height + pad * 2, width + pad * 2))
        localImg = np.zeros((localImgSize, localImgSize))

    padImg[pad:height+pad, pad:width+pad] = image
    localImg_lst = []
    for h in range(pad, height+pad):
        for w in range(pad, width+pad):
            tmpLocalImg = np.array(localImg)
            tmpLocalImg = padImg[h-pad:h+pad, w-pad:w+pad]
            localImg_lst.append(tmpLocalImg)

    return localImg_lst


def main():
    X = tf.placeholder(tf.float32, [None, 72, 72, 3])
    y_ = tf.placeholder(tf.float32, [None])

    #model
    #layer1
    W_conv1 = tf.get_variable("conv1/weight1/weight1", [7,7,3,32])
    b_conv1 = tf.get_variable("conv1/bias1/bias1", [32])
    h_conv1 = tf.nn.relu(cnn.conv2d(X, W_conv1) + b_conv1)
    h_pool1 = cnn.max_pool_2x2(h_conv1)
    #layer2
    W_conv2 = tf.get_variable("conv2/weight2/weight2", [7,7,32,32])
    b_conv2 = tf.get_variable("conv2/bias2/bias2", [32])
    h_conv2 = tf.nn.relu(cnn.conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = cnn.max_pool_2x2(h_conv2)
    #layer3
    W_conv3 = tf.get_variable("conv3/weight3/weight3", [5,5,32,64])
    b_conv3 = tf.get_variable("conv3/bias3/bias3", [64])
    h_conv3 = tf.nn.relu(cnn.conv2d(h_pool2, W_conv3) + b_conv3)
    #layer4
    W_fc4 = tf.get_variable("fc4/weight4/weight4", [18*18*64, 1000])
    b_fc4 = tf.get_variable("fc4/bias4/bias4", [1000])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 18*18*64])
    h_fc4 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc4) + b_fc4)
    #layer5
    W_fc5 = tf.get_variable("fc5/weight5/weight5", [1000, 400])
    b_fc5 = tf.get_variable("fc5/bias5/bias5", [400])
    h_fc5 = tf.nn.relu(tf.matmul(h_fc4, W_fc5) + b_fc5)
    #layer6
    W_fc6 = tf.get_variable("fc6/weight6/weight6", [400, 324])
    b_fc6 = tf.get_variable("fc6/bias6/bias6", [324])
    h_fc6 = tf.nn.relu(tf.matmul(h_fc5, W_fc6) + b_fc6)
    #layer7
    W_fc7 = tf.get_variable("fc7/weight7/weight7", [324, 1])
    b_fc7 = tf.get_variable("fc7/bias7/bias7", [1])
    h_fc7 = tf.nn.relu(tf.matmul(h_fc6, W_fc7) + b_fc7)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model_pixel/model.ckpt")

        img = cv2.imread("../image/original/11_20880.png")
        img = img[:470, :]
        height = img.shape[0]
        width = img.shape[1]
        img_lst = get_local_image(img, 72)
        assert len(img_lst) == height*width
        estImg = np.zeros((height, width))

        i = 0
        for h in range(height):
            for w in range(width):
                output = sess.run(h_fc7, feed_dict={X: img_lst[i].reshape(1, 72, 72, 3)})
                estImg[h][w] = output
                i += 1
                if i%300 == 0:
                    print(i)

        cv2.imwrite("./estimation.png", estImg)
        print("save estimation image")


if __name__ == "__main__":
    main()
