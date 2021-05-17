from typing import Tuple
import numpy as np
import cv2


def resize_two_images(left_img: np.ndarray, right_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Taking the minimum of both heights and widths, then resizing both images to the same size.
    :param left_img: 3D numpy array.
    :param right_img: 3D numpy array.
    :return: two numpy arrays resized to the same size.
    """
    left_height, left_width, _ = left_img.shape
    right_height, right_width, _ = right_img.shape

    desired_height = min(left_height, right_height)
    desired_width = min(left_width, right_width)

    img1 = cv2.resize(left_img, (desired_width, desired_height))
    img2 = cv2.resize(right_img, (desired_width, desired_height))
    return img1, img2


def merge_images(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    merging two
    :param img1: 3D numpy array.
    :param img2: 3D numpy array.
    :param alpha: float, The closer alpha is to 1.0, the more opaque the overlay will be. Similarly, the closer alpha is
    to 0.0, the more transparent the overlay will appear.
    :return: 3D numpy array, a merged image of the two images.
    """
    resized_img1, resized_img2 = resize_two_images(img1.copy(), img2.copy())

    merged_img = cv2.addWeighted(resized_img2, alpha, resized_img1, 1 - alpha, 0)

    return merged_img
