import os
from typing import Tuple, List

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
    :param areas: a list of contour ares.
    :param num_of_bins: the number of intervals.
    :return: a list containing Intervals of class pd.Intervals, The number may be less than num_of_bins because
    of duplicated intervals.
    """
    return list(dict(pd.qcut(areas, num_of_bins, duplicates="drop").value_counts()).keys())


def analyze_default(images: List[np.ndarray], file_names: List[str]):
    """
    In this function we quantify each dimple (internal and external) in each image and save the results for
    each image separately.

    :param images: list of 2D numpy arrays (predictions).
    :param file_names: the name of the image, it will be used for the name of the csv file,example: image_analysis.csv
    :return:
    """
    assert images.__len__() != 0, "Function received empty images list."
    min_limit = 300  # filtering contours with area less than minimum limit.
    num_of_bins = 10
    pixel_to_um = 5
    for index in range(images.__len__()):
        image_analysis = {
            "contour_index": [],
            "contour_type": [],
            "area": [],
            "centroid": [],
            "interval_range": []
        }

        contours, hierarchy = cv2.findContours(images[index], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contours_areas = list(map(lambda x: cv2.contourArea(x), contours))
        max_limit = max(contours_areas)
        _, bin_edges = np.histogram(contours_areas, bins=num_of_bins, range=(min_limit, max_limit))
        for i in range(len(contours)):
            hier = hierarchy[0][i][3]
            cnt = contours[i]
            cnt_area = cv2.contourArea(cnt)
            if min_limit < cnt_area < max_limit:
                image_analysis["contour_index"].append(i)
                if hier == -1:
                    image_analysis["contour_type"].append("external")
                else:
                    image_analysis["contour_type"].append("internal")
                image_analysis["area"].append(cnt_area / pixel_to_um)
                image_analysis["centroid"].append(calcCentroid(cnt))

        intervals = findIntervals(areas=image_analysis["area"], num_of_bins=num_of_bins)

        for area in image_analysis["area"]:
            image_analysis["interval_range"].append(
                str([interval for interval in intervals if area in interval][0]))

        df = pd.DataFrame(image_analysis)
        if not os.path.exists("csv_files/"):
            os.makedirs("csv_files/")
        file_name = file_names[index].split('.')[0]
        df = df.sort_values(by='area', ascending=False)
        df.to_csv(f"csv_files/{file_name}_analysis.csv", index=False)


# def analyze(images: list
#             , show_ex_contours=True
#             , show_in_contours=True
#             , calc_centroid=True
#             , number_of_bins=10
#             , min_limit=300
#             , max_limit=1000000):
#     """
#     This function takes 2D numpy arrays which are black and white images, and finds contours in these images
#     and calculates different properties for each contour
#
#     :param images: 2D numpy array (black and white image).
#     :param show_ex_contours: boolean, if the user wants to quantify the external contours.
#     :param show_in_contours: boolean, if the user wants to quantify the internal contours.
#     :param calc_centroid: boolean, if the user wants to calculate the centroid.
#     :param number_of_bins: the number of intervals.
#     :param min_limit: minimum area in an image.
#     :param max_limit: maximum area in an image.
#     """
#     assert images.__len__() != 0, "Function received empty images list."
#     for image in images:
#         if show_ex_contours and show_in_contours:
#             contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
#             out_p = np.zeros(image.shape + (3,), dtype=np.uint8)
#         for i in range(len(contours)):
#             hier = hierarchy[0][i][3]
#             cnt_color = random_color()
#             cnt = contours[i]
#             cnt_area = cv2.contourArea(cnt)
#             if min_limit < cnt_area < max_limit:
#                 if hier == -1:
#                     # if hier == -1:
#                     cv2.drawContours(external_contours, [cnt], -1, cnt_color, 2)
#                     if calc_centroid:
#                         cx, cy = calcCentroid(cnt)
#                         external_contours = cv2.circle(
#                             external_contours, (cx, cy), radius=3,
#                             color=cnt_color, thickness=-1)
#                 else:
#                     cx, cy = calcCentroid(cnt)
#                     cv2.drawContours(internal_contours, [cnt], -1, cnt_color, 2)
#                     internal_contours = cv2.circle(
#                         internal_contours, (cx, cy), radius=3,
#                         color=cnt_color, thickness=-1)


if __name__ == "__main__":
    mask = cv2.imread("mask.png", 0)
    analyze_default([mask], ["mask.png"])
