import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from annotator.config import AnnotatorConfig
from annotator.constants import D_KEY, DATA_DIR, IMAGE_EXTENTION, P_KEY, Q_KEY, S_KEY
from annotator.exceptions import InputFileTypeError
from annotator.logger import logger
from annotator.utils import (
    get_input_data_type,
    get_path_list,
    load_image,
    load_video,
    save_coordinate,
    save_density_map,
    save_image,
)
from tqdm import tqdm


class DensityAnnotator:
    """DensityAnnotator is a class that labels the position of an object
    with a mouse and calculates a density map for image or video data.

    After annotation, save the raw image, the coordinate of the object,
    and the density map for each target frame.
    """

    def __init__(self, cfg: AnnotatorConfig) -> None:
        """Initialize DensityAnnotator class by AnnotationConfig.

        Args:
            cfg (AnnotatorConfig): config data
        """
        logger.info(f"Loaded config: {cfg}")
        cv2.namedWindow("click annotation points")
        cv2.setMouseCallback("click annotation points", self.mouse_event)
        self.sigma_pow = cfg.sigma_pow
        self.mouse_event_interval = cfg.mouse_event_interval

        # set frame information
        self.video: cv2.VideoCapture
        self.frame: np.ndarray
        self.frame_list: List[np.ndarray]
        self.width: int
        self.height: int
        self.features: np.ndarray
        self.coordinate_matrix: np.ndarray
        self.frame_num = 0

        # set file path
        self.input_file_path: Path
        self.input_file_path_list = get_path_list(DATA_DIR, cfg.path.input_file_path)
        self.save_raw_image_dir = DATA_DIR / cfg.path.save_raw_image_dir
        self.save_annotated_dir = DATA_DIR / cfg.path.save_annotated_dir
        self.save_image_extension = IMAGE_EXTENTION
        self.save_annotated_image_dir = f"{self.save_annotated_dir}/image"
        self.save_annotated_coord_dir = f"{self.save_annotated_dir}/coord"
        self.save_annotated_density_dir = f"{self.save_annotated_dir}/dens"

        # check and create target directory
        os.makedirs(self.save_raw_image_dir, exist_ok=True)
        os.makedirs(self.save_annotated_image_dir, exist_ok=True)
        os.makedirs(self.save_annotated_coord_dir, exist_ok=True)
        os.makedirs(self.save_annotated_density_dir, exist_ok=True)

    def run(self) -> None:
        """Select the data type of the image or video from the extension
        of the input data and execute the annotation.

        Raises:
            InputFileTypeError: input file format not covered
        """
        for file_path in tqdm(self.input_file_path_list, desc="Annotation File Number"):
            # initialization
            self.frame_list = []
            self.features = np.array([], np.uint16)
            # load file
            self.input_file_path = file_path
            data_type = get_input_data_type(self.input_file_path)
            logger.info(f"Annotation Data Type: {data_type}")
            if data_type == "image":
                self.image_annotation()
            elif data_type == "video":
                self.video_annotation()
            else:
                message = f"data_type={data_type} is not defined."
                logger.error(message)
                raise InputFileTypeError(message)

        # end processing
        cv2.destroyAllWindows()

    def annotator_initialization(self) -> None:
        """Initialize coordinate matrix that store the clicked position."""
        self.width = self.frame.shape[1]
        self.height = self.frame.shape[0]

        self.coordinate_matrix = np.zeros((self.width, self.height, 2), dtype="int64")
        for i in range(self.width):
            for j in range(self.height):
                self.coordinate_matrix[i][j] = [i, j]

    def image_annotation(self) -> None:
        """A function to perform annotation on a single image."""
        # load input image
        self.frame = load_image(self.input_file_path)
        self.frame_list.append(self.frame.copy())
        # frame number get from input file name
        self.frame_num = int(
            os.path.splitext(os.path.basename(self.input_file_path))[0]
        )
        # initialize by frame information
        self.annotator_initialization()
        while True:
            # display frame
            cv2.imshow("click annotation points", self.frame)

            # each key operation
            wait_interval = self.mouse_event_interval
            key = cv2.waitKey(wait_interval) & 0xFF
            if key == D_KEY:
                # delete the previous feature point
                self.delete_point()
            elif key == S_KEY:
                # save current annotated data and go to next frame
                self.save_annotated_data()
                wait_interval = self.mouse_event_interval
                break

    def video_annotation(self) -> None:
        """A function to perform annotation on movie.
        This function allow to annotate multiple images cut out
        from the video data at any time.
        """
        # load input video data
        self.video = load_video(self.input_file_path)
        # load first frame and initialize by frame information
        ret, self.frame = self.video.read()
        self.annotator_initialization()

        # read frames at regular intervals and annotate them.
        wait_interval = self.mouse_event_interval
        while ret:
            if wait_interval != 0:
                self.features = np.array([], np.uint16)
                self.frame_num += 1
                # display current frame
                cv2.imshow("click annotation points", self.frame)
                # load next frame and status
                ret, self.frame = self.video.read()

            # each key operation
            key = cv2.waitKey(wait_interval) & 0xFF
            if key == Q_KEY:
                # finish the annotation work
                break
            elif key == P_KEY:
                # pause current frame and start annotation
                wait_interval = 0  # wait until the end of annotation
                self.frame_list.append(self.frame.copy())
                # save raw image
                cv2.imwrite(
                    f"{self.save_raw_image_dir}/{self.frame_num}{self.save_image_extension}",
                    self.frame,
                )
            elif key == D_KEY:
                # delete the previous feature point
                self.delete_point()
            elif key == S_KEY:
                # save current annotated data and go to next frame
                self.save_annotated_data()
                wait_interval = self.mouse_event_interval
        # end processing
        self.video.release()

    def mouse_event(self, event: int, x: int, y: int, flags: int, param: dict) -> None:
        """Select annotated point by left click of mouse

        Args:
            event (int): the type of mouse event
            x (int): x coordinate of the clicked position
            y (int): y coordinate of the clicked position
            flags (int): the type of button or key that was pressed during the mouse event
            param (dict): the value of param set in the third argument of setMouseCallback
        """
        # other than left click
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # draw and add feature point
        cv2.circle(self.frame, (x, y), 4, (0, 0, 255), -1, 8, 0)
        self.add_point(x, y)
        cv2.imshow("click annotation points", self.frame)

    def add_point(self, x: int, y: int) -> None:
        """Add new feature point on stored list

        Args:
            x (int): x coordinate of the clicked position
            y (int): y coordinate of the clicked position
        """
        if self.features.size == 0:
            self.features = np.array([[x, y]], np.uint16)
        else:
            self.features = np.append(self.features, [[x, y]], axis=0).astype(np.uint16)
        self.frame_list.append(self.frame.copy())

    def delete_point(self) -> None:
        """Delete the previous feature point from stored list."""
        if self.features.size > 0:
            self.features = np.delete(self.features, -1, 0)
            self.frame_list.pop()
            self.frame = self.frame_list[-1].copy()
            cv2.imshow("click annotation points", self.frame)

    def calculate_gaussian_kernel(self) -> np.ndarray:
        """Calculate the density map using the Gaussian kernel
        based on the annotated coordinates.

        Returns:
            np.ndarray: calculated density map in numpy format
        """
        kernel = np.zeros((self.width, self.height))

        for point in self.features:
            tmp_coord_matrix = np.array(self.coordinate_matrix)
            point_matrix = np.full((self.width, self.height, 2), point)
            diff_matrix = tmp_coord_matrix - point_matrix
            pow_matrix = diff_matrix * diff_matrix
            norm = pow_matrix[:, :, 0] + pow_matrix[:, :, 1]
            kernel += np.exp(-norm / (2 * self.sigma_pow))

        return kernel.T

    def save_annotated_data(self) -> None:
        """Save coordinate and raw image. There are feature point information."""
        # save image that added annotated point
        save_image(
            Path(
                f"{self.save_annotated_image_dir}/{self.frame_num}{self.save_image_extension}"
            ),
            self.frame,
        )

        # save the coordinates of the annotated point
        save_coordinate(
            Path(f"{self.save_annotated_coord_dir}/{self.frame_num}.csv"), self.features
        )

        # save annotated density map
        annotated_density = self.calculate_gaussian_kernel()
        save_density_map(
            Path(f"{self.save_annotated_density_dir}/{self.frame_num}.npy"),
            annotated_density,
        )
        logger.info(f"Annotated and saved frame number: {self.frame_num}")
