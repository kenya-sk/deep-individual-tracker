import sys
from glob import glob
from typing import List, Tuple

from sklearn.model_selection import train_test_split

from utils import save_dataset_path

RANDOM_SEED = 42


def load_dataset(
    image_directory: str, density_directory: str, file_pattern: str = "*.png"
) -> Tuple[List[str], List[str]]:
    """Load the file name of the data set.

    Args:
        image_directory (str): directory name of image (input)
        density_directory (str): directory name of density map (label)

    Returns:
        Tuple[List[str], List[str]]: tuple with image and label filename pairs
    """
    X_list, y_list = [], []
    file_list = glob(f"{image_directory}/{file_pattern}")
    if len(file_list) == 0:
        sys.stderr.write("Error: Not found input image file")
        sys.exit(1)

    for path in file_list:
        # get label path from input image path
        density_file_name = path.replace(".png", ".npy").split("/")[-1]
        density_path = f"{density_directory}/{density_file_name}"

        # store input and label path
        X_list.append(path)
        y_list.append(density_path)

    return X_list, y_list


def load_multi_date_datasets(
    image_directory: str, density_directory: str, date_list: List[str]
) -> Tuple[List[str], List[str]]:
    """Load input and output pairs based on a list of dates

    Args:
        image_directory (str): directory name of image (input)
        density_directory (str): directory name of density map (label)
        date_list (List[str]): list of dates to be used for splitting

    Returns:
        Tuple[List[str], List[str]]: tuple with image and label filename pairs
    """
    X_multi_list, y_multi_list = [], []
    for date in date_list:
        file_pattern = f"{date}_*.png"
        X_list, y_list = load_dataset(image_directory, density_directory, file_pattern)
        X_multi_list.extend(X_list)
        y_multi_list.extend(y_list)

    return X_multi_list, y_multi_list


def split_dataset(
    X_list: List,
    y_list: List,
    test_size: float,
    save_path_directory: str = None,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """Randomly split the dataset into train, validation, and test.

    Args:
        X_list (List): input image path list
        y_list (List): label path list
        test_size (float): test data size (0.0 - 1.0)
        save_path_directory (str, optional): directory name to save the file name of each dataset. Defaults to None.

    Returns:
        Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
            tuple containing the filenames of train, validation, and test
    """
    # splite dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_list, y_list, test_size=test_size, random_state=RANDOM_SEED
    )
    # split dataset into validation and test
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=RANDOM_SEED
    )

    if save_path_directory is not None:
        save_dataset_path(X_train, y_train, f"{save_path_directory}/train_dataset.csv")
        save_dataset_path(X_valid, y_valid, f"{save_path_directory}/valid_dataset.csv")
        save_dataset_path(X_test, y_test, f"{save_path_directory}/test_dataset.csv")

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def split_dataset_by_date(
    image_directory: str,
    density_directory: str,
    train_date_list: List[str],
    valid_date_list: List[str],
    test_date_list: List[str],
    save_path_directory: str = None,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """split the dataset by date into train, validation, and test.

    Args:
        image_directory (str): directory name of image (input)
        density_directory (str): directory name of density map (label)
        train_date_list (List[str]): date list of training data
        valid_date_list (List[str]): date list of validation data
        test_date_list (List[str]): date list of test data
        save_path_directory (str, optional): directory name to save the file name of each dataset. Defaults to None.

    Returns:
        Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
            tuple containing the filenames of train, validation, and test
    """
    X_train, y_train = load_multi_date_datasets(
        image_directory, density_directory, train_date_list
    )
    X_valid, y_valid = load_multi_date_datasets(
        image_directory, density_directory, valid_date_list
    )
    X_test, y_test = load_multi_date_datasets(
        image_directory, density_directory, test_date_list
    )

    if save_path_directory is not None:
        save_dataset_path(X_train, y_train, f"{save_path_directory}/train_dataset.csv")
        save_dataset_path(X_valid, y_valid, f"{save_path_directory}/valid_dataset.csv")
        save_dataset_path(X_test, y_test, f"{save_path_directory}/test_dataset.csv")

    return X_train, X_valid, X_test, y_train, y_valid, y_test
