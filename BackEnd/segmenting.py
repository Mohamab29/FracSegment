from typing import Union

import cv2
import numpy as np
from tqdm import tqdm
from keras.models import load_model
from skimage.util.shape import view_as_windows
import matplotlib.pyplot as plt

# a global size that we want to resize images to
DESIRED_SIZE = 1024
MODEL_NAME = "model_18-22_20-Mar-2021"


def display(img, title, cmap='gray'):
    """
    :arg img:an image we want to display
    :type title: str
    :arg cmap: using grayscale colo map to show gray scale images
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    plt.title(title)
    ax.imshow(img, cmap=cmap)
    plt.show()


def image_resize(img: np.ndarray, d_size: int = DESIRED_SIZE) -> int:
    """
    the function take an image then resize to a desired size and then add a border to where there us no data
    :param img: an image we want to resize
    :param d_size:the desired size that we want our image to be at
    :return:the image resized with remaining borders being reflected
    """

    old_size = img.shape[:2]  # old_size is in (height, width) format

    # taking the ratio based on the original size
    ratio = float(d_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = d_size - new_size[1]
    delta_h = d_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_img


def split_to_patches(image):
    """
    Spliting a given an image of shape 1024x1024 to 16 patches
    :param: an image we want to create from it patches
    :returns: 16 image patches
    """
    split_images = view_as_windows(image, window_shape=(256, 256), step=256)

    # we use a 4-d shape because that's what our model takes as input
    return np.reshape(split_images, (16, 256, 256, 1))


def patch_back(patches):
    """
    :param patches: an image that is split into 16 images and we want to patch back to one whole image
    :returns: a patched image
    """
    patches = np.reshape(patches, (4, 4, 256, 256))

    patched_image = np.zeros((1024, 1024))
    h, w = 256, 256
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patched_image[i * h:(i + 1) * h, j * w:(j + 1) * w] = patches[i][j]
    return patched_image


def crop_image(img):
    # cropping black borders from a given image
    mask = img > 0
    return img[np.ix_(mask.any(1), mask.any(0))]


def clean_mask(image):
    """
    the function takes as input a mask and we crop the black borders that were produced during the prediction,
    threshold the image to remove unwanted pixels
    :param image: a numpy image
    :returns: a numpy array that is the cleaned mask
    """
    image = (image * 255).astype(np.uint8)
    _, thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY
                                    | cv2.THRESH_OTSU)
    return crop_image(thresh_image)


def load_images(paths: list) -> np.ndarray:
    """
     The function takes a list of image paths and loads them.
     :param paths: a list of paths for images we want to load
     :returns: loaded images of type np.array
    """
    if paths.__len__() == 0:
        return np.array([])

    # a list of the loaded images
    images = []
    # reading the train images or the mask images and converting them to grayscale
    print("Loading images from the given path/s")
    for _, img_path in tqdm(enumerate(paths), total=len(paths)):
        image = cv2.imread(img_path, 0)  # converted to grayscale
        images.append(image.copy())
        image = []
    return np.asarray(images)


def segment(paths: list) -> Union[np.ndarray, list]:
    """
    the main function that handles the image segmentation after the User has chosen to predict selected images.

    :param paths: a list of images we want to segment.
    :returns: a list of predicted masks corresponding to each given image or an empty np.ndarray if an empty list is passed.
    """
    original_images = load_images(paths)
    if original_images.__len__() == 0:
        return original_images

    # after loading the images we would like to resize each image to the desirable size
    resized_images = []
    for img in original_images:
        resized_images.append(image_resize(img))
    resized_images = np.asarray(resized_images)

    # we load the trained model
    model = load_model(f'../assets/models/{MODEL_NAME}.h5')

    # the masks of each image will be contained in this varible
    predicted_masks = []
    for img in resized_images:
        patches = split_to_patches(img)
        prediction = model.predict(x=patches, verbose=1, use_multiprocessing=True)
        patched_img = patch_back(patches=prediction)
        predicted_masks.append(clean_mask(patched_img))

    return predicted_masks
