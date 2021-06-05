import cv2
import numpy as np
import pytest

from BackEnd.segmenting import split_to_patches, patch_back, clean_mask, crop_image, load_images


@pytest.fixture(scope="function")
def original_image():
    image = cv2.imread("./utest_assets/images/3.png")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


@pytest.fixture(scope="function")
def image_for_patch_back():
    image = cv2.imread("./utest_assets/images/image-for-cropping.png")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


@pytest.fixture(scope="module")
def predicted_image():
    image = cv2.imread('./utest_assets/images/3_predicted.png')
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# checking if the types are right and that shapes are as it supposed to be
def test_split_to_patches(original_image):
    # total shape - is the the shape of the image after adding a black border
    patched_image, total_shape = split_to_patches(original_image)
    assert type(patched_image) is np.ndarray
    assert type(total_shape) is tuple
    assert len(patched_image.shape) == 4
    assert len(total_shape) == 2


def test_crop_image(image_for_patch_back):
    cleaned_image = crop_image(image_for_patch_back)
    assert type(cleaned_image) is np.ndarray
    assert len(cleaned_image.shape) == 2


def test_clean_mask(image_for_patch_back):
    cleaned_image = clean_mask(image_for_patch_back)
    assert type(cleaned_image) is np.ndarray
    assert len(cleaned_image.shape) == 2


# checking if the image that is returned has the type
def test_patch_back(predicted_image):
    patched_image, total_shape = split_to_patches(predicted_image)
    patched_back_image = patch_back(patched_image, total_shape)

    assert type(patched_back_image) is np.ndarray
    assert len(patched_back_image.shape) == 2
    np.testing.assert_allclose(predicted_image, clean_mask(patched_back_image))


# checking if we load images successfully
def test_load_images():
    dir_path = "./utest_assets/images/"
    list_of_paths = [
        dir_path + "image-for-cropping.png",
        dir_path + "3.png",
        dir_path + "3_predicted.png"
    ]
    loaded_images = load_images(list_of_paths)
    assert type(loaded_images) is list
    for image in loaded_images:
        assert type(image) is np.ndarray
        assert image.size != 0
