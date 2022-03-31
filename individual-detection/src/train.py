import glob
import logging
import sys
import time
from typing import NoReturn, Tuple

import hydra
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.compat.v1 import ConfigProto, GPUOptions, InteractiveSession
from tensorflow.compat.v1.summary import FileWriter
from tqdm import trange

from model import DensityModel
from utils import (
    apply_masking_on_image,
    get_current_time_str,
    get_elapsed_time_str,
    get_local_data,
    get_masked_index,
    load_image,
    set_tensorboard,
)

# logger setting
current_time = get_current_time_str()
log_path = f"./logs/train_{current_time}.log"
logging.basicConfig(
    filename=log_path,
    level=logging.DEBUG,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(
    image_directory: str,
    density_directory: str,
    mask_path: str,
    input_image_shape: Tuple,
) -> Tuple:
    """_summary_

    Args:
        image_directory (str): _description_
        density_directory (str): _description_
        mask_path (str): _description_
        input_image_shape (Tuple): _description_

    Returns:
        Tuple: _description_
    """
    X_list, y_list = [], []
    file_list = glob.glob("{0}/*.png".format(image_directory))
    if len(file_list) == 0:
        sys.stderr.write("Error: Not found input image file")
        sys.exit(1)

    logger.info("Loading Dataset...")
    for path in file_list:
        image = load_image(path)
        assert (
            image.shape == input_image_shape
        ), f"Invalid image shape. Expected is {input_image_shape}"
        density_file_name = path.replace(".png", ".npy").split("/")[-1]
        density_map = np.load("{0}/{1}".format(density_directory, density_file_name))
        if mask_path is None:
            X_list.append(image)
            y_list.append(density_map)
        else:
            X_list.append(apply_masking_on_image(image, mask_path))
            y_list.append(apply_masking_on_image(density_map, mask_path))

    return np.array(X_list), np.array(y_list)


def split_dataset(X_array: np.array, y_array: np.array, test_size: float) -> Tuple:
    """_summary_

    Args:
        X_array (np.array): _description_
        y_array (np.array): _description_
        test_size (float): _description_

    Returns:
        Tuple: _description_
    """
    # splite dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y_array, test_size=test_size, random_state=42
    )
    # split dataset into validation and test
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def hard_negative_mining(
    X: np.array, y: np.array, loss_array: np.array, prams_dict: dict
) -> Tuple:
    """_summary_

    Args:
        X (np.array): _description_
        y (np.array): _description_
        loss_array (np.array): _description_
        prams_dict (dict): _description_

    Returns:
        Tuple: _description_
    """

    # get index that error is greater than the threshold
    def hard_negative_index(loss_array: np.array, thresh: float) -> np.array:
        """_summary_

        Args:
            loss_array (np.array): _description_
            thresh (float): _description_

        Returns:
            np.array: _description_
        """
        index = np.where(loss_array > thresh)[0]
        return index

    # the threshold is three times the average
    thresh = np.mean(loss_array) * 3
    index = hard_negative_index(loss_array, thresh)
    hard_negative_image_array = np.zeros(
        (len(index), prams_dict["local_image_size"], prams_dict["local_image_size"], 3),
        dtype="uint8",
    )
    hard_negative_label_array = np.zeros((len(index)), dtype="float32")
    for i, hard_index in enumerate(index):
        hard_negative_image_array[i] = X[hard_index]
        hard_negative_label_array[i] = y[hard_index]

    return hard_negative_image_array, hard_negative_label_array


def under_sampling(
    local_iamge_array: np.array, density_array: np.array, thresh: float
) -> Tuple:
    """_summary_

    Args:
        local_iamge_array (np.array): _description_
        density_array (np.array): _description_
        thresh (float): _description_

    Returns:
        Tuple: _description_
    """

    def select(length: int, k: int) -> np.array:
        """Array of boolean which length = length and #True = k

        Args:
            length (int): _description_
            k (int): _description_

        Returns:
            np.array: _description_
        """
        seed = np.arange(length)
        np.random.shuffle(seed)
        return seed < k

    assert len(local_iamge_array) == len(density_array)

    # select all positive samples first
    msk = density_array >= thresh
    # select same number of negative samples with positive samples
    msk[~msk] = select((~msk).sum(), msk.sum())

    return local_iamge_array[msk], density_array[msk]


def horizontal_flip(
    X_train: np.array, y_train: np.array, train_idx: int, params_dict: dict
) -> Tuple:
    """_summary_

    Args:
        X_train (np.array): _description_
        y_train (np.array): _description_
        train_idx (int): _description_
        params_dict (dict): _description_

    Returns:
        Tuple: _description_
    """

    if np.random.rand() < params_dict["flip_prob"]:
        # image apply horizontal flip
        X_train_local, y_train_local = get_local_data(
            X_train[train_idx][:, ::-1, :],
            y_train[train_idx][:, ::-1],
            params_dict,
            is_flip=True,
        )
    else:
        X_train_local, y_train_local = get_local_data(
            X_train[train_idx], y_train[train_idx], params_dict, is_flip=False
        )

    return X_train_local, y_train_local


def train(
    tf_session: InteractiveSession,
    epoch: int,
    model: DensityModel,
    X_train: np.array,
    y_train: np.array,
    params_dict: dict,
    merged,
    writer: FileWriter,
) -> float:
    """_summary_

    Args:
        tf_session (InteractiveSession): _description_
        epoch (int): _description_
        model (DensityModel): _description_
        X_train (np.array): _description_
        y_train (np.array): _description_
        params_dict (dict): _description_
        merged (_type_): _description_
        writer (FileWriter): _description_

    Returns:
        float: _description_
    """

    # initialization of training
    train_loss = 0.0
    hard_negative_image_array = np.zeros(
        (
            1,
            params_dict["local_image_size"],
            params_dict["local_image_size"],
            params_dict["image_channel"],
        ),
        dtype="uint8",
    )
    hard_negative_label_array = np.zeros((1), dtype="float32")

    # one epoch training
    for train_idx in trange(len(X_train), desc=f"Model Training [epoch={epoch}]"):
        # load training local image
        # data augmentation (horizontal flip)
        X_train_local, y_train_local = horizontal_flip(
            X_train, y_train, train_idx, params_dict
        )

        # under sampling
        X_train_local, y_train_local = under_sampling(
            X_train_local, y_train_local, thresh=params_dict["under_sampling_thresh"]
        )

        # hard negative mining
        if hard_negative_label_array.shape[0] > 1:
            X_train_local = np.append(
                X_train_local, hard_negative_image_array[1:], axis=0
            )
            y_train_local = np.append(
                y_train_local, hard_negative_label_array[1:], axis=0
            )
        X_train_local, y_train_local = shuffle(X_train_local, y_train_local)

        # learning by batch
        hard_negative_image_array = np.zeros(
            (
                1,
                params_dict["local_image_size"],
                params_dict["local_image_size"],
                params_dict["image_channel"],
            ),
            dtype="uint8",
        )
        hard_negative_label_array = np.zeros((1), dtype="float32")
        train_n_batches = int(len(X_train_local) / params_dict["batch_size"])
        for train_batch in range(train_n_batches):
            train_start_index = train_batch * params_dict["batch_size"]
            train_end_index = train_start_index + params_dict["batch_size"]

            # training mini batch
            train_diff = tf_session.run(
                model.diff,
                feed_dict={
                    model.X: X_train_local[train_start_index:train_end_index].reshape(
                        -1,
                        params_dict["local_image_size"],
                        params_dict["local_image_size"],
                        params_dict["image_channel"],
                    ),
                    model.y_: y_train_local[train_start_index:train_end_index].reshape(
                        -1, 1
                    ),
                    model.is_training: True,
                    model.keep_prob: params_dict["keep_prob"],
                },
            )
            # update training loss
            train_loss += np.mean(train_diff)

            train_summary, _ = tf_session.run(
                [merged, model.learning_step],
                feed_dict={
                    model.X: X_train_local[train_start_index:train_end_index].reshape(
                        -1,
                        params_dict["local_image_size"],
                        params_dict["local_image_size"],
                        params_dict["image_channel"],
                    ),
                    model.y_: y_train_local[train_start_index:train_end_index].reshape(
                        -1, 1
                    ),
                    model.is_training: True,
                    model.keep_prob: params_dict["keep_prob"],
                },
            )

            # check hard negative sample
            (
                batch_hard_negative_image_array,
                batch_hard_negative_label_array,
            ) = hard_negative_mining(
                X_train_local[train_start_index:train_end_index],
                y_train_local[train_start_index:train_end_index],
                train_diff,
                params_dict,
            )
            # if exist hard negative sample in current batch, append in management array
            if (
                batch_hard_negative_label_array.shape[0] > 0
            ):  # there are hard negative data
                hard_negative_image_array = np.append(
                    hard_negative_image_array, batch_hard_negative_image_array, axis=0
                )
                hard_negative_label_array = np.append(
                    hard_negative_label_array, batch_hard_negative_label_array, axis=0
                )

    # record training summary to TensorBoard
    writer.add_summary(train_summary, epoch)

    # mean train loss per 1 image
    mean_train_loss = train_loss / train_n_batches

    return mean_train_loss


def validation(
    tf_session: InteractiveSession,
    epoch: int,
    model: DensityModel,
    X_valid: np.array,
    y_valid: np.array,
    params_dict: dict,
    merged,
    writer: FileWriter,
) -> float:
    """_summary_

    Args:
        tf_session (InteractiveSession): _description_
        epoch (int): _description_
        model (DensityModel): _description_
        X_valid (np.array): _description_
        y_valid (np.array): _description_
        params_dict (dict): _description_
        merged (_type_): _description_
        writer (FileWriter): _description_

    Returns:
        float: _description_
    """

    valid_loss = 0.0
    for valid_idx in trange(len(X_valid), desc=f"Model Validation [epoch={epoch}]"):
        X_valid_local, y_valid_local = get_local_data(
            X_valid[valid_idx],
            y_valid[valid_idx],
            params_dict["index_h"],
            params_dict["index_w"],
            local_img_size=params_dict["local_image_size"],
        )
        valid_n_batches = int(len(X_valid_local) / params_dict["batch_size"])
        for valid_batch in range(valid_n_batches):
            valid_start_index = valid_batch * params_dict["batch_size"]
            valid_end_index = valid_start_index + params_dict["batch_size"]

            # validate mini batch
            valid_loss_summary, valid_batch_loss = tf_session.run(
                [merged, model.loss],
                feed_dict={
                    model.X: X_valid_local[valid_start_index:valid_end_index].reshape(
                        -1,
                        params_dict["local_image_size"],
                        params_dict["local_image_size"],
                        params_dict["image_channel"],
                    ),
                    model.y_: y_valid_local[valid_start_index:valid_end_index].reshape(
                        -1, 1
                    ),
                    model.is_training: False,
                    model.keep_prob: 1.0,
                },
            )
            # update validation loss
            valid_loss += valid_batch_loss

    # record validation summary to TensorBoard
    writer.add_summary(valid_loss_summary, epoch)

    # mean validation loss per 1 epoch
    mean_valid_loss = valid_loss / valid_n_batches

    return mean_valid_loss


def test(
    tf_session: InteractiveSession,
    model: DensityModel,
    X_test: np.array,
    y_test: np.array,
    params_dict: dict,
    merged,
    writer: FileWriter,
) -> float:
    """_summary_

    Args:
        tf_session (InteractiveSession): _description_
        model (DensityModel): _description_
        X_test (np.array): _description_
        y_test (np.array): _description_
        params_dict (dict): _description_
        merged (_type_): _description_
        writer (FileWriter): _description_

    Returns:
        float: _description_
    """

    test_step = 0
    test_loss = 0.0
    for test_idx in trange(len(X_test), desc="Test Trained Model"):
        X_test_local, y_test_local = get_local_data(
            X_test[test_idx],
            y_test[test_idx],
            params_dict["index_h"],
            params_dict["index_w"],
            local_img_size=params_dict["local_image_size"],
        )
        test_n_batches = int(len(X_test_local) / params_dict["batch_size"])
        for test_batch in range(test_n_batches):
            test_step += 1
            test_start_index = test_batch * params_dict["batch_size"]
            test_end_index = test_start_index + params_dict["batch_size"]

            # test mini batch
            test_summary, test_batch_loss = tf_session.run(
                [merged, model.loss],
                feed_dict={
                    model.X: X_test_local[test_start_index:test_end_index].reshape(
                        -1,
                        params_dict["local_image_size"],
                        params_dict["local_image_size"],
                        params_dict["image_channel"],
                    ),
                    model.y_: y_test_local[test_start_index:test_end_index].reshape(
                        -1, 1
                    ),
                    model.is_training: False,
                    model.keep_prob: 1.0,
                },
            )
            writer.add_summary(test_summary, test_step)
            # update test loss
            test_loss += test_batch_loss

    # mean test loss per 1 epoch
    mean_test_loss = test_loss / test_n_batches

    return mean_test_loss


def model_training(
    X_train: np.array,
    X_valid: np.array,
    X_test: np.array,
    y_train: np.array,
    y_valid: np.array,
    y_test: np.array,
    cfg: dict,
) -> NoReturn:
    """_summary_

    Args:
        X_train (np.array): _description_
        X_valid (np.array): _description_
        X_test (np.array): _description_
        y_train (np.array): _description_
        y_valid (np.array): _description_
        y_test (np.array): _description_
        cfg (dict): _description_

    Returns:
        NoReturn: _description_
    """
    # start TensorFlow session
    tf_config = ConfigProto(
        gpu_options=GPUOptions(
            visible_device_list=cfg["use_gpu_device"],
            per_process_gpu_memory_fraction=cfg["use_memory_rate"],
        )
    )
    tf_session = InteractiveSession(config=tf_config)

    # Tensor Board setting
    summuray_merged, train_writer, valid_writer, test_writer = set_tensorboard(
        cfg["tensorboard_directory"], current_time, tf_session
    )

    # get mask index
    # if you analyze all areas, please set a white image
    index_h, index_w = get_masked_index(cfg["mask_path"], horizontal_flip=False)
    flip_index_h, flip_index_w = get_masked_index(
        cfg["mask_path"], horizontal_flip=True
    )
    cfg["index_h"] = index_h
    cfg["index_w"] = index_w
    cfg["flip_index_h"] = flip_index_h
    cfg["flip_index_w"] = flip_index_w

    # initialization of model variable
    model = DensityModel()
    saver = tf.train.Saver()  # save weight
    # if exist pretrained model, load variable
    ckpt = tf.train.get_checkpoint_state(cfg["pretrained_model_path"])
    if ckpt:
        pretrained_model = ckpt.model_checkpoint_path
        logger.info(f"Load pretraind model: {pretrained_model}")
        saver.restore(tf_session, pretrained_model)
    else:
        logger.debug("Initialize all variable")
        tf.global_variables_initializer().run()

    # training model
    save_model_path = f"{cfg['save_trained_model_directory']}/{current_time}/model.ckpt"
    start_time = time.time()
    valid_loss_list = []
    not_improved_count = 0
    try:
        for epoch in range(cfg["n_epochs"]):
            logger.info(
                f"****************** [epoch: {epoch+1}/{cfg['n_epochs']}] ******************"
            )

            # training
            mean_train_loss = train(
                tf_session,
                epoch,
                model,
                X_train,
                y_train,
                cfg,
                summuray_merged,
                train_writer,
            )

            # validation
            mean_valid_loss = validation(
                tf_session,
                epoch,
                model,
                X_valid,
                y_valid,
                cfg,
                summuray_merged,
                valid_writer,
            )

            # record training results
            valid_loss_list.append(mean_valid_loss)
            logger.info(f"Mean Train Data Loss [per image]: {mean_train_loss}")
            logger.info(f"Mean Valid Data Loss [per image]: {mean_valid_loss}")
            logger.info(f"Elapsed time: {get_elapsed_time_str(start_time)}")

            # check early stopping
            if (epoch > cfg["min_epochs"]) and (
                valid_loss_list[-1] > valid_loss_list[-2]
            ):
                not_improved_count += 1
            else:
                # learning is going well
                not_improved_count = 0
                # save current model
                saver.save(tf_session, save_model_path)
            if not_improved_count >= cfg["early_stopping_epochs"]:
                logger.info(
                    f"early stopping due to not improvement after {cfg['early_stopping_epochs']} epochs."
                )
                break
            logger.info(
                f"not improved count / early stopping epoch: {not_improved_count}/{cfg['early_stopping_epochs']}"
            )

        logger.info(
            "************************** Finish model training *************************"
        )
        # save best model
        saver.save(tf_session, save_model_path)
        logger.info(f'Best model saved in "{save_model_path}"')
        logger.info(
            "**************************************************************************"
        )

        # test trained model
        logger.info(
            "************************** Test trained model *************************"
        )
        mean_test_loss = test(
            tf_session, model, X_test, y_test, cfg, summuray_merged, test_writer
        )

        logger.info(f"Mean Test Data Loss [per image]: {mean_test_loss}")
        logger.info(
            "**************************************************************************"
        )

    # capture Ctrl + C
    except KeyboardInterrupt:
        logger.info('Pressed "Ctrl + C"')
        logger.info("stop training and  save model")
        saver.save(
            tf_session,
            f"{cfg['save_trained_model_directory']}/{current_time}/model.ckpt",
        )

    # close all writer and session
    train_writer.close()
    valid_writer.close()
    test_writer.close()
    tf_session.close()


@hydra.main(config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> NoReturn:
    cfg = OmegaConf.to_container(cfg)
    logger.info(f"Loaded config: {cfg}")

    # loading train, validation and test dataset
    input_image_shape = (cfg["image_height"], cfg["image_width"], cfg["image_channel"])
    X_array, y_array = load_dataset(
        cfg["image_directory"],
        cfg["density_directory"],
        cfg["mask_path"],
        input_image_shape,
    )
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_dataset(
        X_array, y_array, cfg["test_size"]
    )
    logger.info(f"Training Dataset Size: {len(X_train)}")
    logger.info(f"Validation Dataset Size: {len(X_valid)}")
    logger.info(f"Test Dataset Size: {len(X_test)}")

    # training cnn model to predcit densiy map from image
    model_training(X_train, X_valid, X_test, y_train, y_valid, y_test, cfg)


if __name__ == "__main__":
    main()
