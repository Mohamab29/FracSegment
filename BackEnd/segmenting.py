from typing import Tuple
import cv2
import numpy as np
from tqdm import tqdm
from tensorflow import keras
from skimage.util.shape import view_as_windows

# a global size that we want to resize images to
DESIRED_SIZE = 1024
MODEL_NAME = "model"


def split_to_patches(image: np.ndarray, d_size=256) -> Tuple[np.ndarray, tuple]:
    """
    the model takes in 2d array in shape (desired_size,desired_size),so in order to predict a mask for a given images,
    we need to make patches for that image where each patch doesn't overlap with its neighbors.

    we first take an image and then we increase it's shape so that it can be dividable by the desired_size , we then
    use view_as_widows method and make patches.

    :param image: a 2D numpy array.
    :param d_size: the desired size for both the height and width of patch .
    :returns: patches of a given image, and the shape of the image with the new added borders
    """

    original_height, original_width = image.shape
    h_border = original_height - d_size - 1
    w_border = original_width - d_size - 1
    i = 0
    j = 0
    while j <= h_border:
        j += d_size
    while i <= w_border:
        i += d_size

    black_h_border = (j + d_size) - original_height
    black_w_border = (i + d_size) - original_width

    new_image = np.zeros((black_h_border + original_height, black_w_border + original_width))

    new_image[:original_height, :original_width] = image
    patches = view_as_windows(new_image, (d_size, d_size), step=d_size).astype(np.float32)
    rows, cols = patches.shape[0], patches.shape[1]

    patches = patches.reshape((rows * cols, d_size, d_size, 1)) / 255

    return patches, new_image.shape


def patch_back(prediction: np.ndarray, original_shape: tuple, d_size=256) -> np.ndarray:
    """
    When the model makes a prediction for an image it takes patches made form the same image and returns a predicted mask
    with the same amount of patches, so we want to repatch back the prediction into the original shape of the image
    before we made patches from it.

    :param prediction: a numpy array of the shape(number of patches,desired size,desired size,1).
    :param original_shape: a tuple containing the original shape of the image before pachifying.
    :param d_size: the desired size for both the height and width of patch .
    :returns: a numpy array of the mask with the original shape.
    """
    rows, cols = int((prediction.shape[0] * d_size) / original_shape[1]), int(
        (prediction.shape[0] * d_size) / original_shape[0])
    y_pred = prediction.reshape((rows, cols, d_size, d_size))
    pred = np.zeros((original_shape[0], original_shape[1]))
    h, w = d_size, d_size
    for i in range(rows):
        for j in range(cols):
            pred[i * h:(i + 1) * h, j * w:(j + 1) * w] = y_pred[i][j]
    return pred


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


def load_images(paths: list) -> list[np.ndarray]:
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
        del image
    return images


def segment(paths: list, pb_signal) -> np.ndarray:
    """
    the main function that handles the image segmentation after the User has chosen to predict selected images.

    :param pb_signal: Progress bar signal is an emit function from the thread class that we use in order to send the values
    to the progress bar and update it at the end of each iteration.
    :param paths: a list of images we want to segment.
    :returns: a list of predicted masks corresponding to each given image or an empty np.ndarray if an empty list is passed.
    """
    original_images = load_images(paths)
    if original_images.__len__() == 0:
        return original_images

    # we load the trained model
    model = keras.models.load_model(f'../assets/models/{MODEL_NAME}.h5')
    # the masks of each image will be contained in this variable
    predicted_masks = []
    total_images_size = original_images.__len__()
    already_predicted_size = 0
    overall_percentage = 0

    for img in original_images:
        patches, original_shape = split_to_patches(img)
        prediction = model.predict(x=patches, verbose=1, use_multiprocessing=True)
        patched_img = patch_back(prediction, original_shape)
        predicted_masks.append(clean_mask(patched_img))
        already_predicted_size += 1
        overall_percentage = (100 * already_predicted_size) / total_images_size
        pb_signal(overall_percentage)

    return predicted_masks
