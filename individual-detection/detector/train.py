import gc
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
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

from detector.config import TrainConfig, load_config
from detector.constants import (
    CONFIG_DIR,
    DATA_DIR,
    EXECUTION_TIME,
    FLOAT_MAX,
    FRAME_CHANNEL,
    GPU_DEVICE_ID,
    GPU_MEMORY_RATE,
    INPUT_IMAGE_SHAPE,
    LOCAL_IMAGE_SIZE,
    TRAIN_CONFIG_NAME,
)
from detector.exceptions import DatasetSplitTypeError
from detector.index_manager import IndexManager
from detector.logger import logger
from detector.model import DensityModel
from detector.process_dataset import (
    Dataset,
    extract_local_data,
    load_mask_image,
    load_sample,
)
from detector.utils import get_elapsed_time_str, set_tensorboard

tf.compat.v1.disable_eager_execution()


def hard_negative_mining(
    X: np.ndarray, y: np.ndarray, loss_array: np.ndarray, weight: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Hard negative mining is performed based on the error in each sample.

    Args:
        X (np.ndarray): array of local image
        y (np.ndarray): array of density map
        loss_array (np.ndarray): array of each sample loss
        weight (float): weight of hard negative (thresh = weight * mean loss)

    Returns:
        Tuple[np.ndarray, np.ndarray]: tuple of hard negative image and label
    """

    # get index that error is greater than the threshold
    def hard_negative_index(loss_array: np.ndarray, thresh: float) -> np.ndarray:
        """Get the index of the target data from the input sample.

        Args:
            loss_array (np.ndarray): array of each sample loss
            thresh (float): threshold of hard negative

        Returns:
            np.ndarray: array of hard negative index
        """
        index = np.where(loss_array > thresh)[0]
        return index

    # the threshold is three times the average
    thresh = float(np.mean(loss_array) * weight)
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
    local_iamge_array: np.ndarray, density_array: np.ndarray, thresh: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Undersampling to avoid unbalanced labels in the data set.
    The ratio of positive to negative examples should be 1:1.

    Args:
        local_iamge_array (np.ndarray): target local image array
        density_array (np.ndarray): target density value array
        thresh (float): threshold of positive sample

    Returns:
        Tuple[np.ndarray, np.ndarray]: sampled dataset
    """

    def select(length: int, k: int) -> np.ndarray:
        """Array of boolean which length = length and #True = k

        Args:
            length (int): negative sample number
            k (int): positive sample number

        Returns:
            np.ndarray: selected negative sample index
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
    X_image: np.ndarray,
    y_dens: np.ndarray,
    is_flip: bool,
    flip_prob: float,
    index_manager: IndexManager,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get samples of local images to be input to the model from the image and label pairs.

    Args:
        X_image (np.ndarray): target raw image (input)
        y_dens (np.ndarray): target density map (label)
        is_flip (bool): whether apply horizontal flip or not
        flip_prob (float): probability of horizontal flip
        index_manager (IndexManager): index manager class of masked image

    Returns:
        Tuple[np.ndarray, np.ndarray]: local samples array
    """

    if (is_flip) and (np.random.rand() < flip_prob):
        # image apply horizontal flip
        X_train_local_list, y_train_local_list = extract_local_data(
            X_image[:, ::-1, :],
            y_dens[:, ::-1],
            index_manager,
            is_flip=True,
        )
    else:
        X_train_local_list, y_train_local_list = extract_local_data(
            X_image, y_dens, index_manager, is_flip=False
        )

    return np.array(X_train_local_list), np.array(y_train_local_list)


def train(
    tf_session: InteractiveSession,
    epoch: int,
    model: DensityModel,
    X_train: List[Path],
    y_train: List[Path],
    mask_image: np.ndarray,
    index_manager: IndexManager,
    params_dict: TrainConfig,
    summuray_merged: OpsTensor,
    writer: FileWriter,
) -> float:
    """Update and train the parameters of the model using the training data.

    Args:
        tf_session (InteractiveSession): tensorflow session
        epoch (int): epoch number
        model (DensityModel): trained model
        X_train (List[Path]): training input image path List
        y_train (List[Path]): training label path List
        mask_image (np.ndarray): mask image
        index_manager (IndexManager): index manager class of masked image
        params_dict (TrainConfig): config parameters
        summuray_merged (OpsTensor): tensorflow dashboard summury
        writer (FileWriter): tensorflow dashboard writer

    Returns:
        float: training data result (MSE value).
    """

    # initialization of training
    train_loss = 0.0
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
            INPUT_IMAGE_SHAPE,
            mask_image,
            is_rgb=True,
            normalized=True,
        )

        # load training local image
        # data augmentation (horizontal flip)
        X_train_local, y_train_local = get_local_samples(
            X_image, y_dens, True, params_dict.flip_prob, index_manager
        )

        # under sampling
        X_train_local, y_train_local = under_sampling(
            X_train_local, y_train_local, thresh=params_dict.under_sampling_threshold
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
        train_n_batches = int(len(X_train_local) / params_dict.batch_size)
        for train_batch in range(train_n_batches):
            train_start_index = train_batch * params_dict.batch_size
            train_end_index = train_start_index + params_dict.batch_size

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
                    model.dropout_rate: params_dict.dropout_rate,
                },
            )

            # update training loss
            train_loss += float(np.mean(train_diff))

            # check hard negative sample
            (
                batch_hard_negative_image_array,
                batch_hard_negative_label_array,
            ) = hard_negative_mining(
                X_train_local[train_start_index:train_end_index],
                y_train_local[train_start_index:train_end_index],
                train_diff,
                params_dict.hard_negative_mining_weight,
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
    X_valid: List[Path],
    y_valid: List[Path],
    mask_image: np.ndarray,
    index_manager: IndexManager,
    params_dict: TrainConfig,
    summuray_merged: OpsTensor,
    writer: FileWriter,
) -> float:
    """Perform an interim evaluation of the trained model.
    Determine whether overfitting has occurred and whether Early stopping should be performed.

    Args:
        tf_session (InteractiveSession): tensorflow session
        epoch (int): epoch number
        model (DensityModel): trained model
        X_valid (List[Path]): validation input image path List
        y_valid (List[Path]): validation label path List
        mask_image (np.ndarray): mask image
        index_manager (IndexManager): index manager class of masked image
        params_dict (TrainConfig): config parameters
        summuray_merged (OpsTensor): tensorflow dashboard summury
        writer (FileWriter): tensorflow dashboard writer

    Returns:
        float: vaalidation data result (MSE value).
    """

    valid_loss = 0.0
    sample_number = len(X_valid)
    for valid_idx in trange(sample_number, desc=f"Model Validation [epoch={epoch+1}]"):
        # load current index image and label
        X_image, y_dens = load_sample(
            X_valid[valid_idx],
            y_valid[valid_idx],
            INPUT_IMAGE_SHAPE,
            mask_image,
            is_rgb=True,
            normalized=True,
        )

        # load validation local image
        # *Not* apply data augmentation (horizontal flip)
        X_valid_local, y_valid_local = get_local_samples(
            X_image, y_dens, False, params_dict.flip_prob, index_manager
        )

        # under sampling
        X_valid_local, y_valid_local = under_sampling(
            X_valid_local, y_valid_local, thresh=params_dict.under_sampling_threshold
        )

        valid_n_batches = int(len(X_valid_local) / params_dict.batch_size)
        for valid_batch in range(valid_n_batches):
            valid_start_index = valid_batch * params_dict.batch_size
            valid_end_index = valid_start_index + params_dict.batch_size

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
    X_test: List[Path],
    y_test: List[Path],
    mask_image: np.ndarray,
    index_manager: IndexManager,
    params_dict: TrainConfig,
    summuray_merged: OpsTensor,
    writer: FileWriter,
) -> float:
    """Perform a final performance evaluation of the trained model.

    Args:
        tf_session (InteractiveSession): tensorflow session
        model (DensityModel): trained model
        X_test (List[Path]): test input image path List
        y_test (List[Path]): test label path List
        mask_image (np.ndarray): mask image
        index_manager (IndexManager): index manager class of masked image
        params_dict (TrainConfig): config parameters
        summuray_merged (OpsTensor): tensorflow dashboard summury
        writer (FileWriter): tensorflow dashboard writer

    Returns:
        float: test data result (MSE value).
    """

    test_loss = 0.0
    sample_number = len(X_test)
    for test_idx in trange(sample_number, desc="Test Trained Model"):
        # load current index image and label
        X_image, y_dens = load_sample(
            X_test[test_idx],
            y_test[test_idx],
            INPUT_IMAGE_SHAPE,
            mask_image,
            is_rgb=True,
            normalized=True,
        )

        # load test local image
        # *Not* apply data augmentation (horizontal flip)
        X_test_local, y_test_local = get_local_samples(
            X_image, y_dens, False, params_dict.flip_prob, index_manager
        )

        test_n_batches = int(len(X_test_local) / params_dict.batch_size)
        for test_batch in range(test_n_batches):
            test_start_index = test_batch * params_dict.batch_size
            test_end_index = test_start_index + params_dict.batch_size

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
    dataset: Dataset,
    cfg: TrainConfig,
) -> None:
    """Training the model. Perform an interim evaluation using validation data,
    and finally evaluate the learning results with test data.

    Args:
        dataset (Dataset): dataset class for model training
        cfg (TrainConfig): config parameters
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
        DATA_DIR / cfg.tensorboard_directory, EXECUTION_TIME, tf_session
    )

    # get mask index
    # if you analyze all areas, please set a white image
    mask_image = load_mask_image(DATA_DIR / cfg.mask_path)
    index_manager = IndexManager(mask_image)

    # initialization of model variable
    saver = Saver()  # save weight
    # if exist pretrained model, load variable
    ckpt = tf.train.get_checkpoint_state(cfg.pretrained_model_path)
    if ckpt:
        pretrained_model = ckpt.model_checkpoint_path
        logger.info(f"Load pretraind model: {pretrained_model}")
        saver.restore(tf_session, pretrained_model)
    else:
        logger.debug("Initialize all variable")
        global_variables_initializer().run()

    # training model
    save_model_dirc = DATA_DIR / cfg.save_trained_model_directory / f"{EXECUTION_TIME}"
    save_model_dirc.mkdir(parents=True, exist_ok=True)
    save_model_path = save_model_dirc / "model.ckpt"
    start_time = time.time()
    best_valid_loss = FLOAT_MAX
    not_improved_count = 0
    try:
        for epoch in range(cfg.n_epochs):
            logger.info(
                f"****************** [epoch: {epoch+1}/{cfg.n_epochs}] ******************"
            )

            # training
            mean_train_loss = train(
                tf_session,
                epoch,
                model,
                dataset.X_train,
                dataset.y_train,
                mask_image,
                index_manager,
                cfg,
                summuray_merged,
                train_writer,
            )

            # validation
            mean_valid_loss = validation(
                tf_session,
                epoch,
                model,
                dataset.X_valid,
                dataset.y_valid,
                mask_image,
                index_manager,
                cfg,
                summuray_merged,
                valid_writer,
            )

            # record training results
            logger.info(f"Mean Train Data Loss [per image]: {mean_train_loss}")
            logger.info(f"Mean Valid Data Loss [per image]: {mean_valid_loss}")
            logger.info(f"Elapsed time: {get_elapsed_time_str(start_time)}")

            # check early stopping
            if (epoch > cfg.min_epochs) and (mean_valid_loss > best_valid_loss):
                not_improved_count += 1
            else:
                # learning is going well
                not_improved_count = 0
                best_valid_loss = mean_valid_loss
                # save current model
                saver.save(tf_session, save_model_path)

            # excute early stopping
            if not_improved_count >= cfg.early_stopping_patience:
                logger.info(
                    f"early stopping due to not improvement after {cfg.early_stopping_patience} epochs."
                )
                break
            logger.info(
                f"not improved count / early stopping epoch: {not_improved_count}/{cfg.early_stopping_patience}"
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
            dataset.X_test,
            dataset.y_test,
            mask_image,
            index_manager,
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
            DATA_DIR
            / cfg.save_trained_model_directory
            / f"{EXECUTION_TIME}/model.ckpt",
        )

    # close all writer and session
    train_writer.close()
    valid_writer.close()
    test_writer.close()
    tf_session.close()


def main() -> None:
    cfg = load_config(CONFIG_DIR / TRAIN_CONFIG_NAME, TrainConfig)
    logger.info(f"Loaded config: {cfg}")

    # loading train, validation and test dataset
    logger.info("Loading Dataset...")
    save_dataset_path_directory = (
        DATA_DIR / cfg.save_dataset_directory / f"{EXECUTION_TIME}"
    )
    if cfg.dataset_split_type == "random":
        dataset = Dataset(
            DATA_DIR / cfg.image_directory,
            DATA_DIR / cfg.density_directory,
            test_size=cfg.test_size,
            save_path_directory=save_dataset_path_directory,
        )
        dataset.create_random_dataset()
    elif cfg.dataset_split_type == "timeseries":
        dataset = Dataset(
            DATA_DIR / cfg.image_directory,
            DATA_DIR / cfg.density_directory,
            train_date_list=cfg.train_date_list,
            valid_date_list=cfg.validation_date_list,
            test_date_list=cfg.test_date_list,
            save_path_directory=save_dataset_path_directory,
        )
        dataset.create_date_dataset()
    else:
        message = f'dataset_split_type="{cfg.dataset_split_type}" is note defined.'
        logger.error(message)
        raise DatasetSplitTypeError(message)

    logger.info(f"Training Dataset Size: {len(dataset.X_train)}")
    logger.info(f"Validation Dataset Size: {len(dataset.X_valid)}")
    logger.info(f"Test Dataset Size: {len(dataset.X_test)}")

    # training cnn model to predcit densiy map from image
    model_training(dataset, cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
