import os
from typing import Tuple, List, Dict
from segmenting import display
import numpy as np
import cv2
import random
import pandas as pd

"""
When an image has it's mask predicted,A user can then click on the save csv file and the mask (or multiple masks) 
will have it's properties calculated and then saved to a csv file.
Additionally a user can select a mask or multiple masks and send them to the analyze page and in this page, 
The user can choose:
1- The maximum area and/or minimum area to filter dimples.
2- To show the internal contour and/or external contours*.
3- Show the centroid in the image and calculate it
4- number of the bins to divide the area to.
This script in order to handle these options and print them out to a csv file.

*If the user chooses not to show internal and external contours then this means he want an empty black image.  
"""


def find_max_contour(contours: list) -> int:
    """
    the function finds contours,calculates area for each contour and returns the maximum value.

    :param contours: list of contours.
    :return: value of the contour with maximum area.
    """
    return max(list(map(lambda x: cv2.contourArea(x), contours)))


def random_color():
    """returns a tuple of three random integers between 0-255 in order to get a random color each time."""
    return (random.randint(0, 255),
            random.randint(0, 255), random.randint(0, 255))


def calcCentroid(cnt: np.ndarray) -> Tuple[int, int]:
    """
    The function calculates the centroid of a given contour.

    :param cnt: 3D numpy array of the contour shape.
    :return cx,cy: the coordinates of the centroid.
    """
    cx, cy = 0, 0
    try:
        moment = cv2.moments(cnt)
        # Calculate centroid
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
    except ZeroDivisionError:
        print("There was a contour who had an area equal to 0")

    return cx, cy


def findIntervals(areas: list, num_of_bins: int) -> List[pd.Interval]:
    """
    :param areas: a list of contour areas.
    :param num_of_bins: the number of intervals.
    :return: a list containing Intervals of class pd.Intervals, The number may be less than num_of_bins because
    of duplicated intervals.
    """
    intervals = list(dict(pd.qcut(areas, num_of_bins, duplicates="drop", precision=4).value_counts()).keys())
    for i in range(intervals.__len__()):
        intervals[i] = pd.Interval(round(intervals[i].left, 2), round(intervals[i].right, 2), closed="both")
    return intervals


def fitToIntervals(areas: list, num_of_bins: int):
    """
    fitting the area of each contour to the suitable interval.

    :param areas: a list of contour areas.
    :param num_of_bins: the number of intervals.
    :return: a list containing each suitable interval for a given area.
    """
    intervals = findIntervals(areas=areas, num_of_bins=num_of_bins)
    interval_ranges = []
    for area in areas:
        interval_ranges.append(
            str([interval for interval in intervals if area in interval][0]))
    return interval_ranges


def saveAnalysisToCSV(image_analysis: Dict, file_name: str):
    """
    Saving the given dictionary into a csv file with the given file name.

    :param image_analysis: a dictionary containing the properties we wanted to analyze in a prediction.
    :param file_name: the name of the image, it will be used for the name of the csv file,example: image_analysis.csv
    """
    df = pd.DataFrame(image_analysis)
    if not os.path.exists("csv_files/"):
        os.makedirs("csv_files/")
    df = df.sort_values(by='area', ascending=False)
    df.to_csv(f"csv_files/{file_name}_analysis.csv", index=False)


def saveImagesAnalysisToCSV(images_analysis: list, file_names: list):
    """
    If given a list of dictionaries then we save each image analysis with the corresponding file name into csv files.

    :param images_analysis: a list of dictionaries containing the properties we wanted to analyze in a prediction.
    :param file_names: a list of file name that correspond to the given images analysis
    """
    for index in range(images_analysis.__len__()):
        file_name = file_names[index].split('.')[0]
        saveAnalysisToCSV(images_analysis[index], file_name)


def analyze(images: List[np.ndarray]
            , show_ex_contours=True
            , show_in_contours=True
            , calc_centroid=True
            , num_of_bins=10
            , min_limit=300
            , max_limit=1000000) -> Tuple[list, list]:
    """
    This function takes 2D numpy arrays which are black and white images, and finds contours in these images
    and calculates different properties for each contour

    :param images: 2D numpy array (black and white image).
    :param show_ex_contours: boolean, if the user wants to quantify the external contours.
    :param show_in_contours: boolean, if the user wants to quantify the internal contours.
    :param calc_centroid: boolean, if the user wants to calculate the centroid.
    :param num_of_bins: the number of intervals.
    :param min_limit: minimum area in an image.
    :param max_limit: maximum area in an image.
    :returns: two lists, the first list contains the drawn images based on the given options and the second list
     contains image analysis for each given image.
    """

    assert images.__len__() != 0, "Function received empty images list."
    # if show_ex_contours and show_in_contours and calc_centroid and save_to_csv:
    #     analyze_default(images,
    #                     file_names,
    #                     number_of_bins,
    #                     min_limit,
    #                     max_limit)
    images_analysis = []
    image_analysis = {}
    drawn_images = []
    pixel_to_um = 5
    for index in range(images.__len__()):
        contours, hierarchy = cv2.findContours(images[index], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        drawn_image = np.zeros(images[index].shape + (3,), dtype=np.uint8)

        image_analysis["contour_index"] = []
        image_analysis["contour_type"] = []
        image_analysis["area"] = []
        if calc_centroid:
            image_analysis["centroid"] = []
        image_analysis["interval_range"] = []

        for i in range(len(contours)):
            cnt = contours[i]
            hier = hierarchy[0][i][3]
            cnt_area = cv2.contourArea(cnt)
            cnt_color = random_color()
            if min_limit < cnt_area < max_limit:
                if show_in_contours and show_ex_contours:
                    image_analysis["contour_index"].append(i)
                    cv2.drawContours(drawn_image, [cnt], -1, cnt_color, 2)
                    if hier == -1:
                        image_analysis["contour_type"].append("external")
                    else:
                        image_analysis["contour_type"].append("internal")
                    if calc_centroid:
                        cx, cy = calcCentroid(cnt)
                        image_analysis["centroid"].append((cx, cy))
                        drawn_image = cv2.circle(
                            drawn_image, (cx, cy), radius=3,
                            color=cnt_color, thickness=-1)
                    image_analysis["area"].append(cnt_area / pixel_to_um)

                elif show_in_contours and not show_ex_contours:
                    if hier != -1:
                        image_analysis["contour_index"].append(i)
                        cv2.drawContours(drawn_image, [cnt], -1, cnt_color, 2)
                        image_analysis["contour_type"].append("internal")
                        if calc_centroid:
                            cx, cy = calcCentroid(cnt)
                            image_analysis["centroid"].append((cx, cy))
                            drawn_image = cv2.circle(
                                drawn_image, (cx, cy), radius=3,
                                color=cnt_color, thickness=-1)

                        image_analysis["area"].append(cnt_area / pixel_to_um)

                elif show_ex_contours and not show_in_contours:
                    if hier == -1:
                        image_analysis["contour_index"].append(i)
                        cv2.drawContours(drawn_image, [cnt], -1, cnt_color, 2)
                        image_analysis["contour_type"].append("external")

                        if calc_centroid:
                            cx, cy = calcCentroid(cnt)
                            image_analysis["centroid"].append((cx, cy))
                            drawn_image = cv2.circle(
                                drawn_image, (cx, cy), radius=3,
                                color=cnt_color, thickness=-1)
                        image_analysis["area"].append(cnt_area / pixel_to_um)
                else:
                    continue

        if show_in_contours or show_ex_contours:
            image_analysis["interval_range"] = fitToIntervals(areas=image_analysis["area"], num_of_bins=num_of_bins)

        drawn_images.append(drawn_image)
        images_analysis.append(image_analysis)

    return drawn_images, images_analysis


if __name__ == "__main__":
    mask = cv2.imread("mask.png", 0)
    drawn_imgs, analysis = analyze([mask], show_ex_contours=False, calc_centroid=False,
                                   min_limit=1000,
                                   max_limit=200000)
    display(drawn_imgs[0], "Contours")
    saveImagesAnalysisToCSV(images_analysis=analysis, file_names=["mask.png"])
