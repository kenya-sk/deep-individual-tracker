import math
from pathlib import Path
from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession, placeholder, summary
from tensorflow.compat.v1.train import AdamOptimizer

from detector.constants import GPU_DEVICE_ID, GPU_MEMORY_RATE
from detector.exceptions import LoadModelError, TensorShapeError
from detector.logger import logger

tf.compat.v1.disable_eager_execution()


class DensityModel(object):
    """Class defined by 7-layer convolutional neural networks.
    The model takes as input a 3-channel image and outputs the value of the density map corresponding to its center.
    The density map represents the likelihood that each individual is present at that location.
    """

    def __init__(self) -> None:
        # input image
        with tf.compat.v1.name_scope("input"):
            with tf.compat.v1.name_scope("X"):
                self.X = placeholder(tf.float32, [None, 72, 72, 3], name="input")
                _ = summary.image("X", self.X[:, :, :, :], 5)
            # answer image
            with tf.compat.v1.name_scope("y_"):
                self.y_ = placeholder(tf.float32, [None, 1], name="label")
            # status: True(learning) or False(test)
            with tf.compat.v1.name_scope("is_training"):
                self.is_training = placeholder(tf.bool, name="is_training")
            # dropout rate
            with tf.compat.v1.name_scope("keep_prob"):
                self.dropout_rate = placeholder(tf.float32)

        # first layer
        # convolution -> Leaky ReLU -> max pooling
        # input 72x72x3 -> output 36x36x32
        with tf.compat.v1.name_scope("conv1"):
            # 7x7x3 filter
            with tf.compat.v1.name_scope("weight1"):
                w_conv1 = self.__weight_variable([7, 7, 3, 32], None)
                self.__variable_summaries(w_conv1)
                _ = summary.image(
                    "image1",
                    tf.transpose(a=w_conv1, perm=[3, 0, 1, 2])[:, :, :, 0:1],
                    max_outputs=3,
                )
            with tf.compat.v1.name_scope("batchNorm1"):
                conv1 = self.__conv2d(self.X, w_conv1)
                conv1_bn = self.__batch_norm(conv1, [0, 1, 2], 32, self.is_training)
            with tf.compat.v1.name_scope("leakyRelu1"):
                h_conv1 = tf.nn.leaky_relu(conv1_bn)
                self.__variable_summaries(h_conv1)

        with tf.compat.v1.name_scope("pool1"):
            h_pool1 = self.__max_pool_2x2(h_conv1)
            self.__variable_summaries(h_pool1)

        # second layer
        # convolution -> Leaky ReLU -> max pooling
        # input 36x36x32 -> output 18x18x32
        with tf.compat.v1.name_scope("conv2"):
            # 7x7x32 filter
            with tf.compat.v1.name_scope("weight2"):
                w_conv2 = self.__weight_variable([7, 7, 32, 32], None)
                self.__variable_summaries(w_conv2)
                _ = summary.image(
                    "image2",
                    tf.transpose(a=w_conv2, perm=[3, 0, 1, 2])[:, :, :, 0:1],
                    max_outputs=3,
                )
            with tf.compat.v1.name_scope("batchNorm2"):
                conv2 = self.__conv2d(h_pool1, w_conv2)
                conv2_bn = self.__batch_norm(conv2, [0, 1, 2], 32, self.is_training)
            with tf.compat.v1.name_scope("leakyRelu2"):
                h_conv2 = tf.nn.leaky_relu(conv2_bn)
                self.__variable_summaries(h_conv2)

        with tf.compat.v1.name_scope("pool2"):
            h_pool2 = self.__max_pool_2x2(h_conv2)
            self.__variable_summaries(h_pool2)

        # third layer
        # convolution -> Leaky ReLU
        # input 18x18x32 -> output 18x18x64
        with tf.compat.v1.name_scope("conv3"):
            # 5x5x32 filter
            with tf.compat.v1.name_scope("weight3"):
                w_conv3 = self.__weight_variable([5, 5, 32, 64], None)
                self.__variable_summaries(w_conv3)
                _ = summary.image(
                    "image3",
                    tf.transpose(a=w_conv3, perm=[3, 0, 1, 2])[:, :, :, 0:1],
                    max_outputs=3,
                )
            with tf.compat.v1.name_scope("batchNorm3"):
                conv3 = self.__conv2d(h_pool2, w_conv3)
                conv3_bn = self.__batch_norm(conv3, [0, 1, 2], 64, self.is_training)
            with tf.compat.v1.name_scope("leakyRelu3"):
                h_conv3 = tf.nn.leaky_relu(conv3_bn)
                self.__variable_summaries(h_conv3)

        # fourth layer
        # fully connected layer
        # input 18x18x64 -> output 1000
        with tf.compat.v1.name_scope("fc4"):
            with tf.compat.v1.name_scope("weight4"):
                w_fc4 = self.__weight_variable([18 * 18 * 64, 1000], None)
                self.__variable_summaries(w_fc4)
            with tf.compat.v1.name_scope("batchNorm4"):
                h_conv3_flat = tf.reshape(h_conv3, [-1, 18 * 18 * 64])
                fc4 = tf.matmul(h_conv3_flat, w_fc4)
                fc4_bn = self.__batch_norm(fc4, [0], 1000, self.is_training)
            with tf.compat.v1.name_scope("flat4"):
                h_fc4 = tf.nn.leaky_relu(fc4_bn)
                self.__variable_summaries(h_fc4)
            with tf.compat.v1.name_scope("drop4"):
                h_fc4_drop = tf.nn.dropout(h_fc4, rate=self.dropout_rate)

        # fifth layer
        # fully connected layer
        # input 1000 -> output 400
        with tf.compat.v1.name_scope("fc5"):
            with tf.compat.v1.name_scope("weight5"):
                w_fc5 = self.__weight_variable([1000, 400], None)
                self.__variable_summaries(w_fc5)
            with tf.compat.v1.name_scope("batchNorm5"):
                fc5 = tf.matmul(h_fc4_drop, w_fc5)
                fc5_bn = self.__batch_norm(fc5, [0], 400, self.is_training)
            with tf.compat.v1.name_scope("flat5"):
                h_fc5 = tf.nn.leaky_relu(fc5_bn)
                self.__variable_summaries(h_fc5)
            with tf.compat.v1.name_scope("drop5"):
                h_fc5_drop = tf.nn.dropout(h_fc5, rate=self.dropout_rate)

        # sixth layer
        # fully connected layer
        # input 400 -> output 324
        with tf.compat.v1.name_scope("fc6"):
            with tf.compat.v1.name_scope("weight6"):
                w_fc6 = self.__weight_variable([400, 324], None)
                self.__variable_summaries(w_fc6)
            with tf.compat.v1.name_scope("batchNorm6"):
                fc6 = tf.matmul(h_fc5_drop, w_fc6)
                fc6_bn = self.__batch_norm(fc6, [0], 324, self.is_training)
            with tf.compat.v1.name_scope("flat6"):
                h_fc6 = tf.nn.leaky_relu(fc6_bn)
                self.__variable_summaries(h_fc6)
            with tf.compat.v1.name_scope("drop6"):
                h_fc6_drop = tf.nn.dropout(h_fc6, rate=self.dropout_rate)

        # seven layer
        # fully connected layer
        # input 324 -> output 1
        with tf.compat.v1.name_scope("fc7"):
            with tf.compat.v1.name_scope("weight7"):
                w_fc7 = self.__weight_variable([324, 1], None)
                self.__variable_summaries(w_fc7)
            with tf.compat.v1.name_scope("bias7"):
                b_fc7 = self.__bias_variable([1])
                self.__variable_summaries(b_fc7)
            with tf.compat.v1.name_scope("flat7"):
                self.y = tf.nn.relu(tf.matmul(h_fc6_drop, w_fc7) + b_fc7)
                self.__variable_summaries(self.y)

        # output
        summary.histogram("output", self.y)

        # loss function
        with tf.compat.v1.name_scope("loss"):
            self.diff = tf.square(self.y_ - self.y)
            self.loss = tf.reduce_mean(input_tensor=self.diff)
            summary.scalar("loss", self.loss)

        # parameter is default value of tensorflow
        with tf.compat.v1.name_scope("train"):
            self.learning_step = AdamOptimizer(
                learning_rate=0.001,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08,
                use_locking=False,
            ).minimize(self.loss)

    @staticmethod
    def __weight_variable(shape: List[int], name: Optional[str] = None) -> tf.Variable:
        """Initialized weight by He initialization

        Args:
            shape (List[int]): weight shape
            name (str, optional): variable name. Defaults to None.

        Returns:
            tf.Variable: weight variable
        """

        # He initialization
        if len(shape) == 4:
            # convolution layer
            n = shape[1] * shape[2] * shape[3]
        elif len(shape) == 2:
            # fully connected layer
            n = shape[0]
        else:
            message = "Tensor shape must be rank 2 or 4"
            logger.error(message)
            raise TensorShapeError(message)

        stddev = math.sqrt(2 / n)
        initial = tf.random.normal(shape, stddev=stddev, dtype=tf.float32)
        if name is None:
            return tf.Variable(initial)
        else:
            return tf.Variable(initial, name=name)

    @staticmethod
    def __bias_variable(shape: List[int], name: Optional[str] = None) -> tf.Variable:
        """Initialize the bias with a normal distribution (standard deviation: 0.1).

        Args:
            shape (List[int]): shape of bias tensor
            name (str, optional): variable name. Defaults to None.

        Returns:
            tf.Variable: initialized bias
        """

        initial = tf.constant(0.1, shape=shape)
        if name is None:
            return tf.Variable(initial)
        else:
            return tf.Variable(initial, name=name)

    @staticmethod
    def __conv2d(X: tf.Tensor, W: tf.Tensor) -> tf.Tensor:
        """Apply a 2D convolutional layer.

        Args:
            X (tf.Tensor): inputs of counvolutional layer
            W (tf.Tensor): weights of counvolutional layer

        Returns:
            tf.Tensor: output value of counvolutional layer
        """

        return tf.nn.conv2d(input=X, filters=W, strides=[1, 1, 1, 1], padding="SAME")

    @staticmethod
    def __max_pool_2x2(X: tf.Tensor) -> tf.Tensor:
        """Apply 2x2 max pooling layer

        Args:
            X (tf.Tensor): inputs of pooling layer

        Returns:
            tf.Tensor: output value of pooling layer
        """

        return tf.nn.max_pool2d(
            X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )

    @staticmethod
    def __batch_norm(
        X: tf.Tensor, axes: List, shape: int, is_training: bool
    ) -> tf.Tensor:
        """Apply batch normalization on each output layer

        Args:
            X (tf.Tensor): inputs of batch normalization
            axes (List): applied axis list
            shape (int): output shape
            is_training (bool): whether training or not

        Returns:
            tf.Tensor: output value of batch normalization
        """

        if is_training is False:
            return X
        epsilon = 1e-5
        mean, variance = tf.nn.moments(x=X, axes=axes)
        scale = tf.Variable(tf.ones([shape]))
        offset = tf.Variable(tf.zeros([shape]))
        return tf.nn.batch_normalization(X, mean, variance, offset, scale, epsilon)

    @staticmethod
    def __variable_summaries(var: tf.Variable) -> None:
        """Processing variables and it output tensorboard

        Args:
            var (tf.Variable): variable of each layer
        """

        with tf.compat.v1.name_scope("summaries"):
            mean = tf.reduce_mean(input_tensor=var)
            summary.scalar("mean", mean)
            with tf.compat.v1.name_scope("stddev"):
                stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
            summary.scalar("stddev", stddev)
            summary.scalar("max", tf.reduce_max(input_tensor=var))
            summary.scalar("min", tf.reduce_min(input_tensor=var))


def load_model(model_path: Path) -> Tuple:
    """Load trained Convolutional Neural Network model that defined by TensorFlow

    Args:
        model_path (Path): path of trained model

    Returns:
        Tuple: loaded model and tensorflow session
    """

    config = tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(
            visible_device_list=GPU_DEVICE_ID,
            per_process_gpu_memory_fraction=GPU_MEMORY_RATE,
        )
    )
    sess = InteractiveSession(config=config)

    model = DensityModel()
    saver = tf.compat.v1.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt:
        last_model = ckpt.model_checkpoint_path
        logger.info("Load model: {}".format(last_model))
        saver.restore(sess, last_model)
    else:
        message = f'model_path="{model_path}" is not exist.'
        logger.error(message)
        raise LoadModelError(message)

    return model, sess
