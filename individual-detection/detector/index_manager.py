from typing import Tuple

import cv2
import numpy as np
from detector.constants import (
    ANALYSIS_HEIGHT_MAX,
    ANALYSIS_HEIGHT_MIN,
    ANALYSIS_WIDTH_MAX,
    ANALYSIS_WIDTH_MIN,
)


class IndexManager:
    def __init__(self, mask_image: np.ndarray):
        self.index_h, self.index_w = self._get_masked_index(
            mask_image, horizontal_flip=False
        )
        self.flip_index_h, self.flip_index_w = self._get_masked_index(
            mask_image, horizontal_flip=True
        )

    def _get_masked_index(
        self, mask: np.ndarray, horizontal_flip: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Masking an image to get valid index

        Args:
            mask (np.ndarray): binay mask image
            horizontal_flip (bool, optional): Whether to perform data augumentation. Defaults to False.

        Returns:
            Tuple: valid index list (heiht and width)
        """
        # convert gray scale image
        if (len(mask.shape) == 3) and (mask.shape[2] > 1):
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # crop the image to the analysis area
        mask = mask[
            ANALYSIS_HEIGHT_MIN:ANALYSIS_HEIGHT_MAX,
            ANALYSIS_WIDTH_MIN:ANALYSIS_WIDTH_MAX,
        ]

        # index of data augumentation
        if horizontal_flip:
            mask = mask[:, ::-1]

        index = np.where(mask > 0)
        index_h = index[0]
        index_w = index[1]
        assert len(index_h) == len(index_w)

        return index_h, index_w
