import pandas as pd

from BackEnd.analyze_dimples import find_max_area, random_color, saveAnalysisToCSV, calc_centroid, calc_ratio, \
    calc_depth
from BackEnd.analyze_dimples import fitToIntervals
import pytest
import cv2
import os


@pytest.fixture(scope="module")
def predicted_image():
    image = cv2.imread('./utest_assets/images/3_predicted.png')
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# checking if the function returns a tuple and that it contains a number between 0-255
def test_random_color():
    color = random_color()
    assert len(color) == 3
    for c in color:
        assert 0 <= c <= 255


# checking if the value returned is a float number and positive
def test_find_max_area(predicted_image):
    assert len(predicted_image.shape) == 2
    value = find_max_area(predicted_image)
    assert type(value) is float
    assert value > 0


# checking if it returns a list of intervals
def test_fitToIntervals():
    list_of_area = [12, 123.23, 324, 30,
                    50, 43, 100, 200, 89, 454,
                    5.25, 477, 187, 258, 9, 77]
    number_of_bins = 4
    used = set()
    list_of_intervals = fitToIntervals(list_of_area, number_of_bins)
    # taking only the intervals , because list of intervals contains the interval for each corresponding
    # value in list of area
    unique_intervals = [x for x in list_of_intervals if x not in used and (used.add(x) or True)]
    assert type(list_of_intervals) is list
    assert unique_intervals.__len__() == number_of_bins
    assert list_of_area.__len__() == list_of_intervals.__len__()
    # checking if each number is in the intervals
    for i in range(len(list_of_area)):
        intervals = list_of_intervals[i].replace(" ", "").split(',')
        intervals_int = list(
            map(lambda y: float(y.replace("[", "")) if "]" not in y else float(y.replace("]", "")), intervals))
        assert intervals_int[0] <= list_of_area[i] <= intervals_int[1]


@pytest.fixture(scope="module")
def df_for_csv():
    return pd.DataFrame({
        "name": ["testname2", "testname1", "testname3", "testname4"],
        "area": [123, 22, 333, 45]
    })


# checking if the function really saves and does sort
def test_saveAnalysisToCSV(df_for_csv):
    path = "./files/csv_files/"
    saveAnalysisToCSV(df_for_csv, file_name="test", path=".")
    assert os.path.exists(path)
    # checking if the sorting works and it saved all the values
    df_sorted = df_for_csv.sort_values(by='area', ascending=False, ignore_index=True)
    df_from_file = pd.read_csv(path + "test_analysis.csv")
    pd.testing.assert_frame_equal(df_from_file, df_sorted, check_index_type=False)


@pytest.fixture(scope="module")
def contours_from_image(predicted_image):
    # taking some contours
    contours, _ = cv2.findContours(predicted_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    list_of_contours = []
    for i in range(len(contours)):
        cnt = contours[i]
        cnt_area = cv2.contourArea(cnt)
        if 5000 < cnt_area < 1000:
            list_of_contours.append(cnt)
    return list_of_contours


def test_calc_centroid(contours_from_image):
    for cnt in contours_from_image:
        centroid = calc_centroid(cnt)
        assert type(centroid) == tuple
        cx, cy = centroid
        assert type(cx) is int and cx >= 0
        assert type(cy) is int and cy >= 0


def test_calc_ratio(contours_from_image):
    for cnt in contours_from_image:
        result = calc_ratio(cnt)
        assert type(result) == tuple
        ratio, ellipse = result
        assert type(ratio) is float and 0 <= ratio <= 1
        assert type(ellipse) is tuple


@pytest.fixture(scope="module")
def original_image():
    return cv2.imread('./utest_assets/images/3.png')


def test_calc_depth(original_image, contours_from_image):
    for cnt in contours_from_image:
        result = calc_depth(original_image, cnt)
        assert type(result) == tuple
        depth_local, depth_global = result
        assert type(depth_local) is float and depth_local >= 0
        assert type(depth_global) is float and depth_global >= 0
