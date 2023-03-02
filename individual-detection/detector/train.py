import gc
import time
from typing import List, Tuple

import hydra
import numpy as np
import tensorflow as tf
from detector.constants import (
    CONFIG_DIR,
    DATA_DIR,
    EXECUTION_TIME,
    FLOAT_MAX,
    FRAME_CHANNEL,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    GPU_DEVICE_ID,
    GPU_MEMORY_RATE,
    LOCAL_IMAGE_SIZE,
    TRAIN_CONFIG_NAME,
)
from detector.exceptions import DatasetSplitTypeError
from detector.logger import logger
from detector.model import DensityModel
from detector.process_dataset import (
    extract_local_data,
    get_masked_index,
    load_dataset,
    load_mask_image,
    load_sample,
    split_dataset,
    split_dataset_by_date,
)
from detector.utils import get_elapsed_time_str, set_tensorboard
from omegaconf import DictConfig, OmegaConf
from sklearn.utils import shuffle
from tensorflow.compat.v1 import (
    ConfigProto,
    GPUOptions,
    InteractiveSession,
    global_variables_initializer,
)
from tensorflow.compat.v1.summary import FileWriter
from tensorflow.compat.v1.train import Saver
from tensorflow.python.framework.ops import Tensor as OpsTensor
from tqdm import trange


def hard_negative_mining(
    X: np.array, y: np.array, loss_array: np.array, weight: float
) -> Tuple[np.array, np.array]:
    """Hard negative mining is performed based on the error in each sample.

    Args:
        X (np.array): array of local image
        y (np.array): array of density map
        loss_array (np.array): array of each sample loss
        weight (float): weight of hard negative (thresh = weight * mean loss)

    Returns:
        Tuple[np.array, np.array]: tuple of hard negative image and label
    """

    # get index that error is greater than the threshold
    def hard_negative_index(loss_array: np.array, thresh: float) -> np.array:
        """Get the index of the target data from the input sample.

        Args:
            loss_array (np.array): array of each sample loss
            thresh (float): threshold of hard negative

        Returns:
            np.array: array of hard negative index
        """
        index = np.where(loss_array > thresh)[0]
        return index

    # the threshold is three times the average
    thresh = np.mean(loss_array) * weight
    index = hard_negative_index(loss_array, thresh)
    hard_negative_image_array = np.zeros(
        (len(index), LOCAL_IMAGE_SIZE, LOCAL_IMAGE_SIZE, FRAME_CHANNEL),
        dtype="uint8",
    )
    hard_negative_label_array = np.zeros((len(index)), dtype="float32")
    for i, hard_index in enumerate(index):
        hard_negative_image_array[i] = X[hard_index]
        hard_negative_label_array[i] = y[hard_index]

    return hard_negative_image_array, hard_negative_label_array


def under_sampling(
    local_iamge_array: np.array, density_array: np.array, thresh: float
) -> Tuple[np.array, np.array]:
    """Undersampling to avoid unbalanced labels in the data set.
    The ratio of positive to negative examples should be 1:1.

    Args:
        local_iamge_array (np.array): target local image array
        density_array (np.array): target density value array
        thresh (float): threshold of positive sample

    Returns:
        Tuple[np.array, np.array]: sampled dataset
    """

    def select(length: int, k: int) -> np.array:
        """Array of boolean which length = length and #True = k

        Args:
            length (int): negative sample number
            k (int): positive sample number

        Returns:
            np.array: selected negative sample index
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


def get_local_samples(
    X_image: np.array, y_dens: np.array, is_flip: bool, params_dict: dict
) -> Tuple[np.array, np.array]:
    """Get samples of local images to be input to the model from the image and label pairs.

    Args:
        X_image (np.array): target raw image (input)
        y_dens (np.array): target density map (label)
        is_flip (bool): whether apply horizontal flip or not
        params_dict (dict): parameter dictionary

    Returns:
        Tuple[np.array, np.array]: local samples array
    """

    if (is_flip) and (np.random.rand() < params_dict["flip_prob"]):
        # image apply horizontal flip
        X_train_local_list, y_train_local_list = extract_local_data(
            X_image[:, ::-1, :],
            y_dens[:, ::-1],
            params_dict,
            is_flip=True,
        )
    else:
        X_train_local_list, y_train_local_list = extract_local_data(
            X_image, y_dens, params_dict, is_flip=False
        )

    return np.array(X_train_local_list), np.array(y_train_local_list)


def train(
    tf_session: InteractiveSession,
    epoch: int,
    model: DensityModel,
    X_train: List,
    y_train: List,
    mask_image: np.array,
    params_dict: dict,
    summuray_merged: OpsTensor,
    writer: FileWriter,
) -> float:
    """Update and train the parameters of the model using the training data.

    Args:
        tf_session (InteractiveSession): tensorflow session
        epoch (int): epoch number
        model (DensityModel): trained model
        X_train (List): training input image path List
        y_train (List): training label path List
        mask_image (np.array): mask image
        params_dict (dict): parameter dictionary
        summuray_merged (OpsTensor): tensorflow dashboard summury
        writer (FileWriter): tensorflow dashboard writer

    Returns:
        float: training data result (MSE value).
    """

    # initialization of training
    train_loss = 0.0
    input_image_shape = (
        FRAME_HEIGHT,
        FRAME_WIDTH,
        FRAME_CHANNEL,
    )
    hard_negative_image_array = np.zeros(
        (
            1,
            LOCAL_IMAGE_SIZE,
            LOCAL_IMAGE_SIZE,
            FRAME_CHANNEL,
        ),
        dtype="uint8",
    )
    hard_negative_label_array = np.zeros((1), dtype="float32")

    # one epoch training
    sample_number = len(X_train)
    for train_idx in trange(sample_number, desc=f"Model Training [epoch={epoch+1}]"):
        # load current index image and label
        X_image, y_dens = load_sample(
            X_train[train_idx],
            y_train[train_idx],
            input_image_shape,
            mask_image,
            is_rgb=True,
            normalized=True,
        )

        # load training local image
        # data augmentation (horizontal flip)
        X_train_local, y_train_local = get_local_samples(
            X_image, y_dens, True, params_dict
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
                LOCAL_IMAGE_SIZE,
                LOCAL_IMAGE_SIZE,
                FRAME_CHANNEL,
            ),
            dtype="uint8",
        )
        hard_negative_label_array = np.zeros((1), dtype="float32")
        train_n_batches = int(len(X_train_local) / params_dict["batch_size"])
        for train_batch in range(train_n_batches):
            train_start_index = train_batch * params_dict["batch_size"]
            train_end_index = train_start_index + params_dict["batch_size"]

            # training mini batch
            train_diff, train_summary, _ = tf_session.run(
                [model.diff, summuray_merged, model.learning_step],
                feed_dict={
                    model.X: X_train_local[train_start_index:train_end_index].reshape(
                        -1,
                        LOCAL_IMAGE_SIZE,
                        LOCAL_IMAGE_SIZE,
                        FRAME_CHANNEL,
                    ),
                    model.y_: y_train_local[train_start_index:train_end_index].reshape(
                        -1, 1
                    ),
                    model.is_training: True,
                    model.dropout_rate: params_dict["dropout_rate"],
                },
            )

            # update training loss
            train_loss += np.mean(train_diff)

            # check hard negative sample
            (
                batch_hard_negative_image_array,
                batch_hard_negative_label_array,
            ) = hard_negative_mining(
                X_train_local[train_start_index:train_end_index],
                y_train_local[train_start_index:train_end_index],
                train_diff,
                params_dict["hard_negative_weight"],
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

        # release memory
        del X_image, y_dens, X_train_local, y_train_local
        gc.collect()

    # record training summary to TensorBoard
    writer.add_summary(train_summary, epoch)

    # mean train loss per 1 image
    mean_train_loss = train_loss / (sample_number * train_n_batches)

    return mean_train_loss


def validation(
    tf_session: InteractiveSession,
    epoch: int,
    model: DensityModel,
    X_valid: List,
    y_valid: List,
    mask_image: np.array,
    params_dict: dict,
    summuray_merged: OpsTensor,
    writer: FileWriter,
) -> float:
    """Perform an interim evaluation of the trained model.
    Determine whether overfitting has occurred and whether Early stopping should be performed.

    Args:
        tf_session (InteractiveSession): tensorflow session
        epoch (int): epoch number
        model (DensityModel): trained model
        X_valid (List): validation input image path List
        y_valid (List): validation label path List
        mask_image (np.array): mask image
        params_dict (dict): parameter dictionary
        summuray_merged (OpsTensor): tensorflow dashboard summury
        writer (FileWriter): tensorflow dashboard writer

    Returns:
        float: vaalidation data result (MSE value).
    """

    valid_loss = 0.0
    input_image_shape = (
        FRAME_HEIGHT,
        FRAME_WIDTH,
        LOCAL_IMAGE_SIZE,
    )
    sample_number = len(X_valid)
    for valid_idx in trange(sample_number, desc=f"Model Validation [epoch={epoch+1}]"):
        # load current index image and label
        X_image, y_dens = load_sample(
            X_valid[valid_idx],
            y_valid[valid_idx],
            input_image_shape,
            mask_image,
            is_rgb=True,
            normalized=True,
        )

        # load validation local image
        # *Not* apply data augmentation (horizontal flip)
        X_valid_local, y_valid_local = get_local_samples(
            X_image, y_dens, False, params_dict
        )

        # under sampling
        X_valid_local, y_valid_local = under_sampling(
            X_valid_local, y_valid_local, thresh=params_dict["under_sampling_thresh"]
        )

        valid_n_batches = int(len(X_valid_local) / params_dict["batch_size"])
        for valid_batch in range(valid_n_batches):
            valid_start_index = valid_batch * params_dict["batch_size"]
            valid_end_index = valid_start_index + params_dict["batch_size"]

            # validate mini batch
            valid_loss_summary, valid_batch_loss = tf_session.run(
                [summuray_merged, model.loss],
                feed_dict={
                    model.X: X_valid_local[valid_start_index:valid_end_index].reshape(
                        -1,
                        LOCAL_IMAGE_SIZE,
                        LOCAL_IMAGE_SIZE,
                        FRAME_CHANNEL,
                    ),
                    model.y_: y_valid_local[valid_start_index:valid_end_index].reshape(
                        -1, 1
                    ),
                    model.is_training: False,
                    model.dropout_rate: 0.0,
                },
            )
            # update validation loss
            valid_loss += valid_batch_loss

        # release memory
        del X_image, y_dens, X_valid_local, y_valid_local
        gc.collect()

    # record validation summary to TensorBoard
    writer.add_summary(valid_loss_summary, epoch)

    # mean validation loss per 1 epoch
    mean_valid_loss = valid_loss / (sample_number * valid_n_batches)

    return mean_valid_loss


def test(
    tf_session: InteractiveSession,
    model: DensityModel,
    X_test: List,
    y_test: List,
    mask_image: np.array,
    params_dict: dict,
    summuray_merged: OpsTensor,
    writer: FileWriter,
) -> float:
    """Perform a final performance evaluation of the trained model.

    Args:
        tf_session (InteractiveSession): tensorflow session
        model (DensityModel): trained model
        X_test (List): test input image path List
        y_test (List): test label path List
        mask_image (np.array): mask image
        params_dict (dict): parameter dictionary
        summuray_merged (OpsTensor): tensorflow dashboard summury
        writer (FileWriter): tensorflow dashboard writer

    Returns:
        float: test data result (MSE value).
    """

    test_loss = 0.0
    input_image_shape = (
        FRAME_HEIGHT,
        FRAME_WIDTH,
        FRAME_CHANNEL,
    )
    sample_number = len(X_test)
    for test_idx in trange(sample_number, desc="Test Trained Model"):
        # load current index image and label
        X_image, y_dens = load_sample(
            X_test[test_idx],
            y_test[test_idx],
            input_image_shape,
            mask_image,
            is_rgb=True,
            normalized=True,
        )

        # load test local image
        # *Not* apply data augmentation (horizontal flip)
        X_test_local, y_test_local = get_local_samples(
            X_image, y_dens, False, params_dict
        )

        test_n_batches = int(len(X_test_local) / params_dict["batch_size"])
        for test_batch in range(test_n_batches):
            test_start_index = test_batch * params_dict["batch_size"]
            test_end_index = test_start_index + params_dict["batch_size"]

            # test mini batch
            test_summary, test_batch_loss = tf_session.run(
                [summuray_merged, model.loss],
                feed_dict={
                    model.X: X_test_local[test_start_index:test_end_index].reshape(
                        -1,
                        LOCAL_IMAGE_SIZE,
                        LOCAL_IMAGE_SIZE,
                        FRAME_CHANNEL,
                    ),
                    model.y_: y_test_local[test_start_index:test_end_index].reshape(
                        -1, 1
                    ),
                    model.is_training: False,
                    model.dropout_rate: 0.0,
                },
            )
            # update test loss
            test_loss += test_batch_loss

        # release memory
        del X_image, y_dens, X_test_local, y_test_local
        gc.collect()

    # record test summary to TensorBoard
    writer.add_summary(test_summary, 0)

    # mean test loss per 1 epoch
    mean_test_loss = test_loss / (sample_number * test_n_batches)

    return mean_test_loss


def model_training(
    X_train: List,
    X_valid: List,
    X_test: List,
    y_train: List,
    y_valid: List,
    y_test: List,
    cfg: dict,
) -> None:
    """Training the model. Perform an interim evaluation using validation data,
    and finally evaluate the learning results with test data.

    Args:
        X_train (List): training input image path List
        X_valid (List): validation input image path List
        X_test (List): test input image path List
        y_train (List): training label path List
        y_valid (List): validation label path List
        y_test (List): test label path List
        cfg (dict): config dictionary
    """
    # start TensorFlow session
    tf_config = ConfigProto(
        gpu_options=GPUOptions(
            visible_device_list=GPU_DEVICE_ID,
            per_process_gpu_memory_fraction=GPU_MEMORY_RATE,
        )
    )
    tf_session = InteractiveSession(config=tf_config)

    # define model
    model = DensityModel()

    # Tensor Board setting
    summuray_merged, train_writer, valid_writer, test_writer = set_tensorboard(
        str(DATA_DIR / cfg["tensorboard_directory"]), EXECUTION_TIME, tf_session
    )

    # get mask index
    # if you analyze all areas, please set a white image
    mask_image = load_mask_image(cfg["mask_path"])
    index_h, index_w = get_masked_index(mask_image, cfg, horizontal_flip=False)
    flip_index_h, flip_index_w = get_masked_index(mask_image, cfg, horizontal_flip=True)
    cfg["index_h"] = index_h
    cfg["index_w"] = index_w
    cfg["flip_index_h"] = flip_index_h
    cfg["flip_index_w"] = flip_index_w

    # initialization of model variable
    saver = Saver()  # save weight
    # if exist pretrained model, load variable
    ckpt = tf.train.get_checkpoint_state(cfg["pretrained_model_path"])
    if ckpt:
        pretrained_model = ckpt.model_checkpoint_path
        logger.info(f"Load pretraind model: {pretrained_model}")
        saver.restore(tf_session, pretrained_model)
    else:
        logger.debug("Initialize all variable")
        global_variables_initializer().run()

    # training model
    save_model_path = f"{str(DATA_DIR)}/{cfg['save_trained_model_directory']}/{EXECUTION_TIME}/model.ckpt"
    start_time = time.time()
    best_valid_loss = FLOAT_MAX
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
                mask_image,
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
                mask_image,
                cfg,
                summuray_merged,
                valid_writer,
            )

            # record training results
            logger.info(f"Mean Train Data Loss [per image]: {mean_train_loss}")
            logger.info(f"Mean Valid Data Loss [per image]: {mean_valid_loss}")
            logger.info(f"Elapsed time: {get_elapsed_time_str(start_time)}")

            # check early stopping
            if (epoch > cfg["min_epochs"]) and (mean_valid_loss > best_valid_loss):
                not_improved_count += 1
            else:
                # learning is going well
                not_improved_count = 0
                best_valid_loss = mean_valid_loss
                # save current model
                saver.save(tf_session, save_model_path)

            # excute early stopping
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
            tf_session,
            model,
            X_test,
            y_test,
            mask_image,
            cfg,
            summuray_merged,
            test_writer,
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
            f"{str(DATA_DIR)}/{cfg['save_trained_model_directory']}/{EXECUTION_TIME}/model.ckpt",
        )

    # close all writer and session
    train_writer.close()
    valid_writer.close()
    test_writer.close()
    tf_session.close()


@hydra.main(
    config_path=str(CONFIG_DIR), config_name=TRAIN_CONFIG_NAME, version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg)
    logger.info(f"Loaded config: {cfg}")

    # loading train, validation and test dataset
    logger.info("Loading Dataset...")
    save_dataset_path_directory = (
        f"{DATA_DIR}/cfg['save_dataset_path_directory']/{EXECUTION_TIME}"
    )
    if cfg["dataset_split_type"] == "random":
        X_list, y_list = load_dataset(
            str(DATA_DIR / cfg["image_directory"]),
            str(DATA_DIR / cfg["density_directory"]),
        )
        X_train, X_valid, X_test, y_train, y_valid, y_test = split_dataset(
            X_list,
            y_list,
            cfg["test_size"],
            save_path_directory=save_dataset_path_directory,
        )
    elif cfg["dataset_split_type"] == "timeseries":
        X_train, X_valid, X_test, y_train, y_valid, y_test = split_dataset_by_date(
            str(DATA_DIR / cfg["image_directory"]),
            str(DATA_DIR / cfg["density_directory"]),
            cfg["train_date_list"],
            cfg["valid_date_list"],
            cfg["test_date_list"],
            save_path_directory=save_dataset_path_directory,
        )
    else:
        message = f'dataset_split_type="{cfg["dataset_split_type"]}" is note defined.'
        logger.error(message)
        raise DatasetSplitTypeError(message)

    logger.info(f"Training Dataset Size: {len(X_train)}")
    logger.info(f"Validation Dataset Size: {len(X_valid)}")
    logger.info(f"Test Dataset Size: {len(X_test)}")

    # training cnn model to predcit densiy map from image
    model_training(X_train, X_valid, X_test, y_train, y_valid, y_test, cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)