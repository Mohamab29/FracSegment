import os
from typing import Tuple, List, Dict
import numpy as np
import cv2
import random
import pandas as pd
import matplotlib as plt
from matplotlib import pyplot
import seaborn as sns

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


def find_max_area(image: np.ndarray) -> float:
    """
    the function finds contours,calculates area for each contour and returns the maximum value.

    :param image: image 2D array of contours.
    :return: value of the contour with maximum area.
    """
    contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
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


def saveAnalysisToCSV(image_analysis: Dict, file_name: str, path: str):
    """
    Saving the given dictionary into a csv file with the given file name.

    :param image_analysis: a dictionary containing the properties we wanted to analyze in a prediction.
    :param file_name: the name of the image, it will be used for the name of the csv file,example: image_analysis.csv
    :param path: folder path.
    """
    df = pd.DataFrame(image_analysis)
    if not os.path.exists(f"{path}/files/csv_files/"):
        os.makedirs(f"{path}/files/csv_files/")
    df = df.sort_values(by='area', ascending=False)
    df.to_csv(f"{path}/files/csv_files/{file_name}_analysis.csv", index=False)


def saveImagesAnalysisToCSV(images_analysis: list, file_names: list, path: str):
    """
    If given a list of dictionaries then we save each image analysis with the corresponding file name into csv files.

    :param images_analysis: a list of dictionaries containing the properties we wanted to analyze in a prediction.
    :param file_names: a list of file name that correspond to the given images analysis
    :param path: folder path.
    """
    for index in range(images_analysis.__len__()):
        file_name = file_names[index].split('.')[0]
        saveAnalysisToCSV(images_analysis[index], file_name, path)


def saveImagesToHistPlots(images_analysis: list, file_names: list, path: str, num_of_bins: int = 15):
    """
    If given a list of dictionaries, we send each dictionary to the createAndSaveHistPlot function and then we create
    a graph for each image and then we save it with the corresponding file name into a png file.

    :param images_analysis: a list of dictionaries containing the properties we analyzed in a prediction.
    :param num_of_bins: the number of intervals.
    :param file_names: a list of file name that correspond to the given images analysis
    :param path: folder path.
    """
    for index in range(images_analysis.__len__()):
        file_name = file_names[index].split('.')[0]
        createAndSaveHistPlot(images_analysis[index], num_of_bins, file_name, path)


def createAndSaveHistPlot(image_analysis: dict, num_of_bins: int, file_name: str, path: str = "."):
    """
    The function takes a dictionary of an image analysis that contains properties of a single image,
    and makes a histogram plot based on the area as the x axis and the frequency as the y-axis.


    :param image_analysis: a dictionary containing the properties we wanted to analyze in a prediction.
    :param num_of_bins: the number of intervals.
    :param file_name: string, the name of the image, it will be used for the name of the csv file,example: image_graph.png
    :param path:string,folder path,default is "." .

    """
    intervals = findIntervals(image_analysis["area"], num_of_bins=num_of_bins)
    # left_intervals  = [interval.left for interval in intervals]

    df = pd.DataFrame({
        "area": image_analysis["area"],
        "intervals": image_analysis["interval_range"]
    })
    ax_dims = (15, 15)
    fig, ax = pyplot.subplots(figsize=ax_dims)
    ax = sns.histplot(ax=ax, data=df, x="intervals", multiple="dodge",
                      shrink=0.5, stat="frequency", color="skyblue")
    ax.set_xlabel('Intervals in μm',
                  fontsize='xx-large')
    ax.set_ylabel('Frequency',
                  fontsize='x-large')
    x = range(0, len(intervals))
    ax.set_xticks(x)
    ax.set_xticklabels(labels=intervals, rotation=45, size=15)
    y = range(0, 31)
    ax.set_yticks(y)
    ax.margins(0.2, 0.2)
    ax.autoscale(enable=True, axis="x", tight=True)
    fig.tight_layout()
    if not os.path.exists(f"{path}/files/graphs/"):
        os.makedirs(f"{path}/files/graphs/")
    ax.figure.savefig(f"{path}/files/graphs/{file_name}_graph.png", dpi=120)


# noinspection DuplicatedCode
def analyze(images: dict
            , flags: dict
            , min_max_values: dict
            , num_of_bins=15) -> Tuple[dict, dict]:
    """
    Analyze black and white images, and finds contours in these images
    and calculates different properties for each contour

    :param images: contains values as np.ndarray which are the images and the keys are the image names.
    :param flags: a dictionary that contains the following flags:<br>
        1- show_ex_contours: boolean, if the user wants to quantify the external contours.<br>
        2- show_in_contours: boolean, if the user wants to quantify the internal contours.<br>
        3- calc_centroid: boolean, if the user wants to calculate the centroid.<br>
    :param min_max_values: a dictionary that contains requested min and max value for each image.
    :param num_of_bins: the number of intervals.
    :returns: two dictionaries, the first dictionary values are the drawn images based on the given options and <br>
        the second dictionary values are image analysis for each given image,the keys are image names.
    """

    assert images.__len__() != 0, "Function received empty images list."

    images_analysis = {}
    drawn_images = {}
    # pixel_to_um = 5
    images_names = list(images.keys())
    for name in images_names:

        contours, hierarchy = cv2.findContours(images[name], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        drawn_image = np.zeros(images[name].shape + (3,), dtype=np.uint8)

        image_analysis = {"contour_index": [], "contour_type": [], "area": []}
        if flags["calc_centroid"]:
            image_analysis["centroid"] = []
        image_analysis["interval_range"] = []

        min_limit, max_limit = min_max_values[name]

        for i in range(len(contours)):
            cnt = contours[i]
            hier = hierarchy[0][i][3]
            cnt_area = cv2.contourArea(cnt)
            cnt_color = random_color()
            if min_limit < cnt_area < max_limit:

                if flags["show_in_contours"] and flags["show_ex_contours"]:
                    image_analysis["contour_index"].append(i)
                    cv2.drawContours(drawn_image, [cnt], -1, cnt_color, 2)
                    if hier == -1:
                        image_analysis["contour_type"].append("external")
                    else:
                        image_analysis["contour_type"].append("internal")
                    if flags["calc_centroid"]:
                        cx, cy = calcCentroid(cnt)
                        image_analysis["centroid"].append((cx, cy))
                        cv2.circle(
                            drawn_image, (cx, cy), radius=3,
                            color=cnt_color, thickness=-1)
                    image_analysis["area"].append(cnt_area)

                elif flags["show_in_contours"] and not flags["show_ex_contours"]:
                    if hier != -1:
                        image_analysis["contour_index"].append(i)
                        cv2.drawContours(drawn_image, [cnt], -1, cnt_color, 2)
                        image_analysis["contour_type"].append("internal")
                        if flags["calc_centroid"]:
                            cx, cy = calcCentroid(cnt)
                            image_analysis["centroid"].append((cx, cy))
                            cv2.circle(
                                drawn_image, (cx, cy), radius=3,
                                color=cnt_color, thickness=-1)

                        image_analysis["area"].append(cnt_area)

                elif flags["show_ex_contours"] and not flags["show_in_contours"]:
                    if hier == -1:
                        image_analysis["contour_index"].append(i)
                        cv2.drawContours(drawn_image, [cnt], -1, cnt_color, 2)
                        image_analysis["contour_type"].append("external")

                        if flags["calc_centroid"]:
                            cx, cy = calcCentroid(cnt)
                            image_analysis["centroid"].append((cx, cy))
                            cv2.circle(
                                drawn_image, (cx, cy), radius=3,
                                color=cnt_color, thickness=-1)
                        image_analysis["area"].append(cnt_area)
                else:
                    continue

        if flags["show_in_contours"] or flags["show_ex_contours"]:
            image_analysis["interval_range"] = fitToIntervals(areas=image_analysis["area"], num_of_bins=num_of_bins)

        drawn_images[name] = drawn_image
        images_analysis[name] = image_analysis

    return drawn_images, images_analysis
