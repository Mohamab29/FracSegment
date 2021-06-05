import cv2
import numpy as np
import pytest
from BackEnd.merge_images import resize_two_images
from BackEnd.merge_images import merge_images


@pytest.fixture(scope="module")
def colored_image():
    return cv2.imread('./utest_assets/images/colored-image-merge.png')


@pytest.fixture(scope="module")
def original_image():
    return cv2.imread('./utest_assets/images/image-merge.png')


# we check if the function resize two images returns two images with the same size
def test_resize_two_images(original_image, colored_image):
    img1, img2 = resize_two_images(original_image, colored_image)
    assert img1.shape == img2.shape


# we check if the function returns a 3d numpy array
def test_merge_images(original_image, colored_image):
    img = merge_images(original_image, colored_image)
    assert type(img) == np.ndarray
    assert len(img.shape) == 3
