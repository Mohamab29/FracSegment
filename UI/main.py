import os

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QDesktopWidget, QFrame
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from ui_main import Ui_MainWindow
from functools import partial
from PIL.ImageQt import ImageQt
from BackEnd.segmenting import segment
from BackEnd.analyze_dimples import analyze, saveImagesAnalysisToCSV, find_max_area
import cv2
import io
from PIL import Image
from PyQt5.QtCore import QBuffer


def changeButtonToDisableStyle(btn):
    btn.setStyleSheet("QPushButton {\n"
                      "    color: rgb(43, 49, 56);\n"
                      "    background-color: rgb(33, 37, 43);\n"
                      "    border: 2px outset rgb(37, 40, 45);\n"
                      "    border-radius: 3px;\n"
                      "}\n"
                      "QPushButton:hover {\n"
                      "    color: rgb(85, 170, 255);\n"
                      "}")


def changeButtonToEnableStyle(btn):
    btn.setStyleSheet("QPushButton {\n"
                      "    color: rgb(140, 166, 179);\n"
                      "    background-color: rgb(33, 37, 43);\n"
                      "    border: 2px outset rgb(70, 76, 85);\n"
                      "    border-radius: 3px;\n"
                      "}\n"
                      "QPushButton:hover {\n"
                      "    color: rgb(85, 170, 255);\n"
                      "    background-color: rgb(42, 47, 54);\n"
                      "}")


def changeChackBoxToEnableStyle(check_box):
    check_box.setStyleSheet("color: rgb(255, 255, 255);")


def changeChackBoxToDisableStyle(check_box):
    check_box.setStyleSheet("color: rgb(32, 35, 41);")


def toggleCheckBoxAndChangeStyle(pair):
    for p in pair:
        check_box = p[0]
        term = p[1]
        if not term:
            check_box.setEnabled(False)
            changeChackBoxToDisableStyle(check_box)
        else:
            check_box.setEnabled(True)
            changeChackBoxToEnableStyle(check_box)


def toggleButtonAndChangeStyle(pair):
    for p in pair:
        btn = p[0]
        term = p[1]
        if not term:
            btn.setEnabled(False)
            changeButtonToDisableStyle(btn)
        else:
            btn.setEnabled(True)
            changeButtonToEnableStyle(btn)


def setListItemItemStyle(item):
    font = QtGui.QFont()
    font.setBold(True)
    font.setWeight(75)
    item.setFont(font)


def openImage(path):
    image = Image.open(path, 'r')
    image.show()


def showDialog(title, message, icon):
    msg_box = QMessageBox()
    msg_box.setIcon(icon)
    msg_box.setText(message)
    msg_box.setWindowTitle(title)
    msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

    qr = msg_box.frameGeometry()
    cp = QDesktopWidget().availableGeometry().center()
    qr.moveCenter(cp)
    msg_box.move(qr.center())

    return_value = msg_box.exec()
    if return_value == QMessageBox.Ok:
        return True
    else:
        return False


def isAtLeastOneItemChecked(widget_list):
    for index in range(widget_list.count()):
        if widget_list.item(index).checkState() == 2:
            return True
    return False


def isAtLeastOneItemNotChecked(widget_list):
    for index in range(widget_list.count()):
        if widget_list.item(index).checkState() == 0:
            return True
    return False


def isNoItemChecked(widget_list):
    for index in range(widget_list.count()):
        if widget_list.item(index).checkState() == 2:
            return False
    return True


def numOfCheckedItems(widget_list):
    counter = 0
    for index in range(widget_list.count()):
        if widget_list.item(index).checkState() == 2:
            counter = counter + 1
    return counter


def imageLabelFrame(label, frame_shape=0, frame_shadow=0, line_width=0):
    label.setFrameShape(frame_shape)
    label.setFrameShadow(frame_shadow)
    label.setLineWidth(line_width)


def evnImageListItemDoubleClickedPagePredict(dic, item):
    if item:
        openImage(dic[item.text()])


def evnImageListItemDoubleClicked(dic, item):
    if item:
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        dic[item.text()].save(buffer, item.text().split('.')[-1])
        pil_im = Image.open(io.BytesIO(buffer.data()))
        pil_im.show()


def convertCvImage2QtImage(img, mode):
    """
    Converting an image to a pix map. from a numpy array to and image object to a Pixmap

    :returns: a Pixmap object
    """
    if mode == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return QtGui.QPixmap.fromImage(ImageQt(Image.fromarray(img).convert(mode)))


def convertCvImage2QtImageRGB(img, mode):
    """
    Converting an image to a pix map. from a numpy array to and image object to a Pixmap

    :returns: a Pixmap object
    """
    if mode == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = QtGui.QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QtGui.QImage.Format_RGB888)
    pix = QtGui.QPixmap(image.copy())
    return pix


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.animation = QtCore.QPropertyAnimation(self.ui.frame_left_menu, b"minimumWidth")
        self.imageListPathDict = {}
        self.PredictedImagesPixMap = {}
        self.PredictedImagesNpArray = {}
        # In order to save in the prediction page to CSV, we need to send to the analyze function
        # default flags
        self.default_flags = {
            'show_in_contours': True,
            'show_ex_contours': True,
            'calc_centroid': True
        }

        self.imagesForCalculationNpArray = {}
        self.imagesForCalculationPixMap = {}

        # saving the changes everytime we analyze images
        self.imagesAnalyse = {}
        self.imagesDrawn = {}

        self.imagesMaxValues = {}
        self.imagesMinValues = {}
        self.currentItemClickedNameCalcPage = ''
        self.setActions()
        self.center()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def setActions(self):
        self.ui.btn_page_predict.clicked.connect(self.evnPagePredictClicked)
        self.ui.btn_page_results.clicked.connect(self.evnPageResultsClicked)
        self.ui.btn_page_help.clicked.connect(self.evnPageHelpClicked)
        self.ui.btn_page_calculation.clicked.connect(self.evnPageCalculationClicked)
        self.ui.btn_toggle.clicked.connect(self.evnBtnToggleClicked)

        self.ui.btn_predict_page_load_images.clicked.connect(self.evnLoadImagesButtonClicked)
        self.ui.btn_predict_page_clear_images.clicked.connect(self.evnClearImagesButtonClickedPagePredict)
        self.ui.btn_predict_page_check_all.clicked.connect(self.evnCheckAllButtonClickedPagePredict)
        self.ui.btn_predict_page_uncheck_all.clicked.connect(self.evnUncheckAllButtonClickedPagePredict)
        self.ui.btn_predict_page_delete_selected_images.clicked.connect(
            self.evnDeleteSelectedImagesButtonClickedPagePredict)
        self.ui.btn_predict_page_predict.clicked.connect(self.evnPredictButtonClicked)

        self.ui.images_predict_page_import_list.itemClicked.connect(self.evnImageListItemClickedPagePredict)
        self.ui.images_predict_page_import_list.itemDoubleClicked.connect(
            partial(evnImageListItemDoubleClickedPagePredict, self.imageListPathDict))
        self.ui.images_predict_page_import_list.currentItemChanged.connect(
            partial(self.evnCurrentItemChangedPagePredict, self.ui.label_predict_page_selected_picture))

        self.ui.btn_results_page_clear_images.clicked.connect(self.evnClearImagesButtonClickedPageResults)
        self.ui.btn_results_page_check_all.clicked.connect(self.evnCheckAllButtonClickedPageResults)
        self.ui.btn_results_page_uncheck_all.clicked.connect(self.evnUncheckAllButtonClickedPageResults)
        self.ui.btn_results_page_delete_selected_images.clicked.connect(
            self.evnDeleteSelectedImagesButtonClickedPageResults)
        self.ui.btn_results_page_save_images.clicked.connect(self.evnSaveSelectedImagesButtonClickedPageResults)
        self.ui.btn_results_page_custom_calculation.clicked.connect(self.evnCustomCalculationButtonClicked)
        self.ui.btn_results_page_save_csvs.clicked.connect(self.evnSaveCsvsButtonClickedPageResults)
        self.ui.btn_results_page_save_images_and_csvs.clicked.connect(self.evnSaveImageAndCsvsButtonClickedPageResults)

        self.ui.images_results_page_import_list.itemClicked.connect(self.evnImageListItemClickedPageResults)
        self.ui.images_results_page_import_list.itemDoubleClicked.connect(
            partial(evnImageListItemDoubleClicked, self.PredictedImagesPixMap))
        self.ui.images_results_page_import_list.currentItemChanged.connect(
            partial(self.evnCurrentItemChangedPageResults, self.ui.label_results_page_selected_picture))

        self.ui.btn_calculation_page_clear_images.clicked.connect(self.evnClearImagesButtonClickedPageCalculation)
        self.ui.btn_calculation_page_send.clicked.connect(self.evnSendButtonClickedPageCalculation)
        self.ui.btn_calculation_page_check_all.clicked.connect(self.evnCheckAllButtonClickedPageCalculation)
        self.ui.btn_calculation_page_uncheck_all.clicked.connect(self.evnUncheckAllButtonClickedPageCalculation)
        self.ui.btn_calculation_page_delete_selected_images.clicked.connect(
            self.evnDeleteSelectedImagesButtonClickedPageCalculation)
        self.ui.btn_calculation_page_save_images.clicked.connect(self.evnSaveImagesButtonClickedPageCalculation)
        self.ui.btn_calculation_page_save_csvs.clicked.connect(self.evnSaveCsvsButtonClickedPageCalculation)

        self.ui.images_calculation_page_import_list.itemClicked.connect(self.evnImageListItemClickedPageCalculation)
        self.ui.images_calculation_page_import_list.itemDoubleClicked.connect(
            partial(evnImageListItemDoubleClicked, self.imagesForCalculationPixMap))
        self.ui.images_calculation_page_import_list.currentItemChanged.connect(
            partial(self.evnCurrentItemChangedPageCalculation, self.ui.label_calculate_page_selected_picture))

        self.ui.frame_calculation_page_modifications_options_max_spin_box.valueChanged.connect(
            self.evnChangeMaxValuePageCalculation)
        self.ui.frame_calculation_page_modifications_options_min_spin_box.valueChanged.connect(
            self.evnChangeMinValuePageCalculation)

    def evnPredictButtonClicked(self):
        checked_items = []
        import_list = self.ui.images_predict_page_import_list

        for index in range(import_list.count()):
            if import_list.item(index).checkState() == 2:
                # add check if the image already predicted.
                list_item = import_list.item(index)
                checked_items.append(self.imageListPathDict[list_item.text()])

        segmented_images = segment(checked_items)
        segmented_images_size = segmented_images.__len__()

        if segmented_images_size:
            for index in range(segmented_images_size):
                segmented_image = segmented_images[index]
                splited_image_name = checked_items[index].split('/')[-1].split('.')
                image_name = f"{splited_image_name[0]}_predicted.{splited_image_name[1]}"

                self.PredictedImagesPixMap[image_name] = convertCvImage2QtImage(segmented_image, "L")
                self.PredictedImagesNpArray[image_name] = segmented_image

                self.addImageNameToList(image_name, self.ui.images_results_page_import_list)

            buttons_tuple = [
                (self.ui.btn_results_page_clear_images, True),
                (self.ui.btn_results_page_uncheck_all, True),
                (self.ui.btn_results_page_delete_selected_images, True),
                (self.ui.btn_results_page_save_images, True),
                (self.ui.btn_results_page_custom_calculation, True),
                (self.ui.btn_results_page_save_csvs, True),
                (self.ui.btn_results_page_save_images_and_csvs, True)
            ]

            toggleButtonAndChangeStyle(buttons_tuple)
            import_list = self.ui.images_results_page_import_list
            selected_result_list_size = len(import_list.selectedItems())

            if not selected_result_list_size:
                label = self.ui.label_results_page_selected_picture
                label.setPixmap(QtGui.QPixmap(convertCvImage2QtImage(segmented_images[len(segmented_images) - 1], "L")))
                imageLabelFrame(label, QFrame.StyledPanel, QFrame.Sunken, 3)
                # label = self.ui.label_results_page_selected_picture
                # imageLabelFrame(label, 0, 0, 0)
                # label.setText("Please select image to view on the screen.")
            self.updateNumOfImagesPageResults(import_list)

    def evnCustomCalculationButtonClicked(self):

        results_page_list = self.ui.images_results_page_import_list

        checked_min_max_values = {}

        for index in range(results_page_list.count()):
            if results_page_list.item(index).checkState() == 2:
                list_item = results_page_list.item(index)
                list_item_name = list_item.text()
                list_item_calc_name = list_item_name.replace('_predicted', '_calculated')

                self.imagesForCalculationNpArray[list_item_calc_name] = \
                    self.PredictedImagesNpArray[list_item_name]

                self.imagesMaxValues[list_item_calc_name] = find_max_area(
                    self.PredictedImagesNpArray[list_item_name])
                self.imagesMinValues[
                    list_item_calc_name] = self.ui.frame_calculation_page_modifications_options_min_spin_box.value()
                checked_min_max_values[list_item_calc_name] = (
                    self.imagesMinValues[list_item_calc_name], self.imagesMaxValues[list_item_calc_name])

        checked_calculated_items_size = self.imagesForCalculationNpArray.__len__()

        if checked_calculated_items_size:
            self.imagesDrawn, self.imagesAnalyse = analyze(self.imagesForCalculationNpArray, self.default_flags,
                                                           checked_min_max_values)
            nparray_images = self.imagesForCalculationNpArray.keys()
            names = []
            for name in nparray_images:
                self.imagesForCalculationPixMap[name] = convertCvImage2QtImageRGB(self.imagesDrawn[name].copy(),
                                                                                  "RGB")
                self.addImageNameToList(name, self.ui.images_calculation_page_import_list)
                names.append(name)

            buttons_tuple = [
                (self.ui.btn_calculation_page_save_images, True),
                (self.ui.btn_calculation_page_save_csvs, True),
                (self.ui.btn_calculation_page_delete_selected_images, True),
                (self.ui.btn_calculation_page_clear_images, True),
                (self.ui.btn_calculation_page_send, True),
                (self.ui.btn_calculation_page_uncheck_all, True)

            ]

            check_box_tuple = [
                (self.ui.frame_calculation_page_modifications_options_min_spin_box, True),
                (self.ui.frame_calculation_page_modifications_options_max_spin_box, True),
                (self.ui.frame_calculation_page_modifications_options_max_label, True),
                (self.ui.frame_calculation_page_modifications_options_min_label, True),
                (self.ui.check_box_show_and_calculate_centroid, True),
                (self.ui.check_box_show_external_contures, True),
                (self.ui.check_box_show_internal_contures, True)]

            toggleButtonAndChangeStyle(buttons_tuple)
            toggleCheckBoxAndChangeStyle(check_box_tuple)
            import_list = self.ui.images_calculation_page_import_list

            selected_calculation_list_size = len(import_list.selectedItems())

            if not selected_calculation_list_size:
                last_image = self.imagesDrawn[names[-1]]
                label = self.ui.label_calculate_page_selected_picture
                label.setPixmap(QtGui.QPixmap(convertCvImage2QtImageRGB(last_image.copy(), "RGB")))
                imageLabelFrame(label, QFrame.StyledPanel, QFrame.Sunken, 3)
            self.updateNumOfImagesPageCalculation(import_list)
        names = []

    def evnSaveCsvsButtonClickedPageResults(self):
        predicted_images_nparray = {}

        checked_min_max_values = {}
        default_min_value = self.ui.frame_calculation_page_modifications_options_min_spin_box.value()
        default_max_value = self.ui.frame_calculation_page_modifications_options_max_spin_box.value()
        results_page_list = self.ui.images_results_page_import_list

        for index in range(results_page_list.count()):
            if results_page_list.item(index).checkState() == 2:
                list_item = results_page_list.item(index)
                list_item_name = list_item.text()

                predicted_images_nparray[list_item_name] = self.PredictedImagesNpArray[list_item_name]
                checked_min_max_values[list_item_name] = (default_min_value, default_max_value)

        path = QFileDialog.getExistingDirectory(self, "Choose Folder")

        if path and predicted_images_nparray:
            _, images_analysis = analyze(predicted_images_nparray, self.default_flags, checked_min_max_values)
            saveImagesAnalysisToCSV(list(images_analysis.values()), list(images_analysis.keys()), path)

    def addImageNameToList(self, image_name, list):
        list_item = QtWidgets.QListWidgetItem()
        list_item.setCheckState(QtCore.Qt.Checked)
        list_item.setText(image_name)
        setListItemItemStyle(list_item)
        list.addItem(list_item)

    def evnSaveSelectedImagesButtonClickedPageResults(self):

        checked_items = {}

        for index in range(self.ui.images_results_page_import_list.count()):
            if self.ui.images_results_page_import_list.item(index).checkState() == 2:
                list_item = self.ui.images_results_page_import_list.item(index)
                checked_items[list_item.text()] = self.PredictedImagesPixMap[list_item.text()]

        res = QFileDialog.getExistingDirectory(self, "Choose Folder")

        if res and checked_items:
            if not os.path.exists(f"{res}/files/predicted_images/"):
                os.makedirs(f"{res}/files/predicted_images/")

            for image_name in checked_items:
                pixmap_image = checked_items[image_name]
                image_path = f"{res}/files/predicted_images/{image_name}"
                image_type = image_name.split('.')[-1]
                pixmap_image.save(image_path, image_type)

    def evnSaveImageAndCsvsButtonClickedPageResults(self):
        predicted_images_pixmap = {}
        predicted_images_nparray = {}

        checked_min_max_values = {}
        default_min_value = self.ui.frame_calculation_page_modifications_options_min_spin_box.value()
        default_max_value = self.ui.frame_calculation_page_modifications_options_max_spin_box.value()

        results_page_list = self.ui.images_results_page_import_list

        for index in range(results_page_list.count()):
            if results_page_list.item(index).checkState() == 2:
                list_item = results_page_list.item(index)
                list_item_name = list_item.text()

                predicted_images_pixmap[list_item_name] = self.PredictedImagesPixMap[list_item_name]
                predicted_images_nparray[list_item_name] = self.PredictedImagesNpArray[list_item_name]
                checked_min_max_values[list_item_name] = (default_min_value, default_max_value)

        path = QFileDialog.getExistingDirectory(self, "Choose Folder")

        if path and predicted_images_nparray and predicted_images_pixmap:

            _, images_analysis = analyze(predicted_images_nparray, self.default_flags, checked_min_max_values)
            saveImagesAnalysisToCSV(list(images_analysis.values()), list(images_analysis.keys()), path)

            if not os.path.exists(f"{path}/files/predicted_images/"):
                os.makedirs(f"{path}/files/predicted_images/")

            for image_name in predicted_images_pixmap:
                pixmap_image = predicted_images_pixmap[image_name]
                image_path = f"{path}/files/predicted_images/{image_name}"
                image_type = image_name.split('.')[-1]
                pixmap_image.save(image_path, image_type)

    def evnCurrentItemChangedPagePredict(self, label, item):
        if item:
            label.setPixmap(QtGui.QPixmap(self.imageListPathDict[item.text()]))
            imageLabelFrame(label, QFrame.StyledPanel, QFrame.Sunken, 3)

    def evnCurrentItemChangedPageResults(self, label, item):
        if item:
            label.setPixmap(self.PredictedImagesPixMap[item.text()])
            imageLabelFrame(label, QFrame.StyledPanel, QFrame.Sunken, 3)

    def evnCurrentItemChangedPageCalculation(self, label, item):
        if item:
            image = convertCvImage2QtImageRGB(self.imagesDrawn[item.text()], "RGB")
            label.setPixmap(image)
            imageLabelFrame(label, QFrame.StyledPanel, QFrame.Sunken, 3)
            self.currentItemClickedNameCalcPage = item.text()

    def evnChangeMaxValuePageCalculation(self):
        self.imagesMaxValues[
            self.currentItemClickedNameCalcPage] = self.ui.frame_calculation_page_modifications_options_max_spin_box.value()
        self.ui.frame_calculation_page_modifications_options_max_spin_box.setValue(self.imagesMaxValues[
                                                                                       self.currentItemClickedNameCalcPage])

    def evnChangeMinValuePageCalculation(self):
        self.imagesMinValues[
            self.currentItemClickedNameCalcPage] = self.ui.frame_calculation_page_modifications_options_min_spin_box.value()
        self.ui.frame_calculation_page_modifications_options_min_spin_box.setValue(self.imagesMinValues[
                                                                                       self.currentItemClickedNameCalcPage])

    def sharedTermsPagePredict(self):
        widget_list = self.ui.images_predict_page_import_list

        self.updateNumOfImagesPagePredict(widget_list)

        if not isAtLeastOneItemChecked(widget_list):
            toggleButtonAndChangeStyle([(self.ui.btn_predict_page_uncheck_all, False)])
        else:
            toggleButtonAndChangeStyle([(self.ui.btn_predict_page_uncheck_all, True)])

        if isAtLeastOneItemNotChecked(widget_list):
            toggleButtonAndChangeStyle([(self.ui.btn_predict_page_check_all, True)])
        else:
            toggleButtonAndChangeStyle([(self.ui.btn_predict_page_check_all, False)])

        if numOfCheckedItems(widget_list):
            toggleButtonAndChangeStyle([(self.ui.btn_predict_page_predict, True)])
            toggleButtonAndChangeStyle([(self.ui.btn_predict_page_delete_selected_images, True)])
        else:
            toggleButtonAndChangeStyle([(self.ui.btn_predict_page_predict, False)])
            toggleButtonAndChangeStyle([(self.ui.btn_predict_page_delete_selected_images, False)])

    def sharedTermsPageResults(self):
        widget_list = self.ui.images_results_page_import_list

        self.updateNumOfImagesPageResults(widget_list)

        if not isAtLeastOneItemChecked(widget_list):
            toggleButtonAndChangeStyle([(self.ui.btn_results_page_uncheck_all, False)])
        else:
            toggleButtonAndChangeStyle([(self.ui.btn_results_page_uncheck_all, True)])

        if isAtLeastOneItemNotChecked(widget_list):
            toggleButtonAndChangeStyle([(self.ui.btn_results_page_check_all, True)])
        else:
            toggleButtonAndChangeStyle([(self.ui.btn_results_page_check_all, False)])

        if numOfCheckedItems(widget_list):
            toggleButtonAndChangeStyle([(self.ui.btn_results_page_delete_selected_images, True)])
        else:
            toggleButtonAndChangeStyle([(self.ui.btn_results_page_delete_selected_images, False)])

        if not numOfCheckedItems(widget_list):
            buttons_tuple = [(self.ui.btn_results_page_custom_calculation, False),
                             (self.ui.btn_results_page_save_images, False),
                             (self.ui.btn_results_page_save_csvs, False),
                             (self.ui.btn_results_page_save_images_and_csvs, False),
                             ]

            toggleButtonAndChangeStyle(buttons_tuple)
        else:
            buttons_tuple = [(self.ui.btn_results_page_custom_calculation, True),
                             (self.ui.btn_results_page_save_images, True),
                             (self.ui.btn_results_page_save_csvs, True),
                             (self.ui.btn_results_page_save_images_and_csvs, True),
                             ]

            toggleButtonAndChangeStyle(buttons_tuple)

    def sharedTermsPageCalculation(self):
        widget_list = self.ui.images_calculation_page_import_list

        self.updateNumOfImagesPageCalculation(widget_list)

        if not isAtLeastOneItemChecked(widget_list):
            toggleButtonAndChangeStyle([(self.ui.btn_calculation_page_uncheck_all, False)])
        else:
            toggleButtonAndChangeStyle([(self.ui.btn_calculation_page_uncheck_all, True)])

        if isAtLeastOneItemNotChecked(widget_list):
            toggleButtonAndChangeStyle([(self.ui.btn_calculation_page_check_all, True)])
        else:
            toggleButtonAndChangeStyle([(self.ui.btn_calculation_page_check_all, False)])

        if numOfCheckedItems(widget_list):
            toggleButtonAndChangeStyle([(self.ui.btn_calculation_page_delete_selected_images, True)])
        else:
            toggleButtonAndChangeStyle([(self.ui.btn_calculation_page_delete_selected_images, False)])

        if not numOfCheckedItems(widget_list):
            buttons_tuple = [(self.ui.btn_calculation_page_clear_images, False),
                             (self.ui.btn_calculation_page_save_images, False),
                             (self.ui.btn_calculation_page_save_csvs, False),
                             (self.ui.btn_calculation_page_send, False)
                             ]

            check_box_tuple = [
                (self.ui.frame_calculation_page_modifications_options_min_spin_box, False),
                (self.ui.frame_calculation_page_modifications_options_max_spin_box, False),
                (self.ui.frame_calculation_page_modifications_options_max_label, False),
                (self.ui.frame_calculation_page_modifications_options_min_label, False),
                (self.ui.check_box_show_and_calculate_centroid, False),
                (self.ui.check_box_show_external_contures, False),
                (self.ui.check_box_show_internal_contures, False)]

            toggleButtonAndChangeStyle(buttons_tuple)
            toggleCheckBoxAndChangeStyle(check_box_tuple)
        else:
            buttons_tuple = [(self.ui.btn_calculation_page_clear_images, True),
                             (self.ui.btn_calculation_page_save_images, True),
                             (self.ui.btn_calculation_page_save_csvs, True),
                             (self.ui.btn_calculation_page_send, True)
                             ]

            check_box_tuple = [
                (self.ui.frame_calculation_page_modifications_options_min_spin_box, True),
                (self.ui.frame_calculation_page_modifications_options_max_spin_box, True),
                (self.ui.frame_calculation_page_modifications_options_max_label, True),
                (self.ui.frame_calculation_page_modifications_options_min_label, True),
                (self.ui.check_box_show_and_calculate_centroid, True),
                (self.ui.check_box_show_external_contures, True),
                (self.ui.check_box_show_internal_contures, True)]

            toggleButtonAndChangeStyle(buttons_tuple)
            toggleCheckBoxAndChangeStyle(check_box_tuple)

    def evnImageListItemClickedPagePredict(self):
        self.sharedTermsPagePredict()

    def evnImageListItemClickedPageResults(self):
        self.sharedTermsPageResults()

    def evnImageListItemClickedPageCalculation(self, item):
        self.ui.frame_calculation_page_modifications_options_max_spin_box.setValue(
            self.imagesMaxValues[item.text()])
        self.ui.frame_calculation_page_modifications_options_min_spin_box.setValue(
            self.imagesMinValues[item.text()])
        self.currentItemClickedNameCalcPage = item.text()
        self.sharedTermsPageCalculation()

    def evnPagePredictClicked(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.frame_predict_page)

    def evnPageResultsClicked(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.frame_results_page)

    def evnPageHelpClicked(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.frame_help_page)

    def evnPageCalculationClicked(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.frame_calculation_page)

    def evnBtnToggleClicked(self):
        self.toggleMenu(150, True)

    def toggleMenu(self, max_width, enable):
        if enable:
            # GET WIDTH
            width = self.ui.frame_left_menu.width()
            max_extend = max_width
            standard = 50

            # SET MAX WIDTH
            if width == 50:
                width_extended = max_extend
                self.ui.btn_page_predict.setText('Predict')
                self.ui.btn_page_results.setText('Results')
                self.ui.btn_page_calculation.setText('Calculation')
                self.ui.btn_page_help.setText('Help')

            else:
                width_extended = standard
                self.ui.btn_page_predict.setText('')
                self.ui.btn_page_results.setText('')
                self.ui.btn_page_calculation.setText('')
                self.ui.btn_page_help.setText('')

            # ANIMATION
            self.animation.setDuration(10)
            self.animation.setStartValue(width)
            self.animation.setEndValue(width_extended)
            self.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
            self.animation.start()

    def evnLoadImagesButtonClicked(self):
        file_to_open = "Image Files (*.png *.jpg *.bmp *.tiff)"
        res = QFileDialog.getOpenFileNames(None, "Open File", "/", file_to_open)

        if len(res[0]) > 0:
            self.renderInputPictureList(res[0])

    def renderInputPictureList(self, pictures_input_path):
        new_image_flag = False
        for i in range(len(pictures_input_path)):
            if pictures_input_path[i].split('/')[-1] not in self.imageListPathDict:
                new_image_flag = True
                self.imageListPathDict[pictures_input_path[i].split('/')[-1]] = pictures_input_path[i]
                list_item = QtWidgets.QListWidgetItem()
                list_item.setCheckState(QtCore.Qt.Checked)
                list_item.setText(pictures_input_path[i].split('/')[-1])
                setListItemItemStyle(list_item)
                self.ui.images_predict_page_import_list.addItem(list_item)

        if new_image_flag:
            buttons_tuple = [(self.ui.btn_predict_page_predict, True),
                             (self.ui.btn_predict_page_clear_images, True),
                             (self.ui.btn_predict_page_uncheck_all, True),
                             (self.ui.btn_predict_page_delete_selected_images, True)]
            toggleButtonAndChangeStyle(buttons_tuple)

        # if len(self.ui.images_predict_page_import_list.selectedItems()) == 0:
        # imageLabelFrame(label, 0, 0, 0)
        # label.setText("Please select image to view on the screen.")
        label = self.ui.label_predict_page_selected_picture
        last_picture_name = pictures_input_path[len(pictures_input_path) - 1].split('/')[-1]
        label.setPixmap(QtGui.QPixmap(self.imageListPathDict[last_picture_name]))
        imageLabelFrame(label, QFrame.StyledPanel, QFrame.Sunken, 3)

        self.updateNumOfImagesPagePredict(self.ui.images_predict_page_import_list)

    def evnDeleteSelectedImagesButtonClickedPagePredict(self):
        checked_items = []

        if showDialog('Delete the selected images', 'Are you sure?', QMessageBox.Information):
            for index in range(self.ui.images_predict_page_import_list.count()):
                if self.ui.images_predict_page_import_list.item(index).checkState() == 2:
                    list_item = self.ui.images_predict_page_import_list.item(index)
                    checked_items.append(list_item)

            for item in checked_items:
                self.ui.images_predict_page_import_list.takeItem(self.ui.images_predict_page_import_list.row(item))
                self.imageListPathDict.pop(item.text())

            self.sharedTermsPagePredict()

            if not len(self.ui.images_predict_page_import_list):
                label = self.ui.label_predict_page_selected_picture
                label.setText("Please load and select image.")
                imageLabelFrame(label, 0, 0, 0)
                toggleButtonAndChangeStyle([(self.ui.btn_predict_page_clear_images, False)])

    def evnDeleteSelectedImagesButtonClickedPageResults(self):
        checked_items = []

        if showDialog('Delete the selected images', 'Are you sure?', QMessageBox.Information):
            for index in range(self.ui.images_results_page_import_list.count()):
                if self.ui.images_results_page_import_list.item(index).checkState() == 2:
                    list_item = self.ui.images_results_page_import_list.item(index)
                    checked_items.append(list_item)

            for item in checked_items:
                self.ui.images_results_page_import_list.takeItem(self.ui.images_results_page_import_list.row(item))
                self.PredictedImagesPixMap.pop(item.text())

            self.sharedTermsPageResults()

            if not len(self.ui.images_results_page_import_list):
                label = self.ui.label_results_page_selected_picture
                label.setText("Please predict images first.")
                imageLabelFrame(label, 0, 0, 0)
                toggleButtonAndChangeStyle([(self.ui.btn_results_page_clear_images, False)])

    def evnDeleteSelectedImagesButtonClickedPageCalculation(self):
        checked_items = []

        if showDialog('Delete the selected images', 'Are you sure?', QMessageBox.Information):
            for index in range(self.ui.images_calculation_page_import_list.count()):
                if self.ui.images_calculation_page_import_list.item(index).checkState() == 2:
                    list_item = self.ui.images_calculation_page_import_list.item(index)
                    checked_items.append(list_item)

            for item in checked_items:
                self.ui.images_calculation_page_import_list.takeItem(
                    self.ui.images_calculation_page_import_list.row(item))
                self.imagesForCalculationNpArray.pop(item.text())
                self.imagesForCalculationPixMap.pop(item.text())

            self.sharedTermsPageCalculation()

            if not len(self.ui.images_calculation_page_import_list):
                label = self.ui.label_calculate_page_selected_picture
                label.setText("Please predict images first.")
                imageLabelFrame(label, 0, 0, 0)
                toggleButtonAndChangeStyle([(self.ui.btn_calculation_page_clear_images, False)])

    def evnSaveImagesButtonClickedPageCalculation(self):
        checked_images = []

        calculation_items_list = self.ui.images_calculation_page_import_list

        for index in range(calculation_items_list.count()):
            item = calculation_items_list.item(index)
            if item.checkState() == 2:
                list_item = item
                list_item_name = list_item.text()
                checked_images.append(list_item_name)

        path = QFileDialog.getExistingDirectory(self, "Choose Folder")

        if path and checked_images:
            if not os.path.exists(f"{path}/files/drawn_images/"):
                os.makedirs(f"{path}/files/drawn_images/")
            for image_name in checked_images:
                image_path = f"{path}/files/drawn_images/{image_name}"
                cv2.imwrite(image_path, self.imagesDrawn[image_name])

    def evnSaveCsvsButtonClickedPageCalculation(self):

        calculated_images_save = {}
        calculation_page_list = self.ui.images_calculation_page_import_list

        for index in range(calculation_page_list.count()):
            if calculation_page_list.item(index).checkState() == 2:
                list_item = calculation_page_list.item(index)
                list_item_name = list_item.text()
                calculated_images_save[list_item_name] = self.imagesAnalyse[list_item_name]

        path = QFileDialog.getExistingDirectory(self, "Choose Folder")

        if path and calculated_images_save:
            saveImagesAnalysisToCSV(list(calculated_images_save.values()), list(calculated_images_save.keys()), path)

    def evnClearImagesButtonClickedPagePredict(self):
        if self.imageListPathDict and showDialog('Clear all images', 'Are you sure?', QMessageBox.Information):
            self.ui.images_predict_page_import_list.clear()
            self.imageListPathDict = {}
            imageLabelFrame(self.ui.label_predict_page_selected_picture, 0, 0, 0)
            self.ui.label_predict_page_selected_picture.setText("Please load and select image.")

            buttons_tuple = [(self.ui.btn_predict_page_uncheck_all, False),
                             (self.ui.btn_predict_page_check_all, False),
                             (self.ui.btn_predict_page_predict, False),
                             (self.ui.btn_predict_page_delete_selected_images, False),
                             (self.ui.btn_predict_page_clear_images, False)]

            toggleButtonAndChangeStyle(buttons_tuple)
            self.updateNumOfImagesPagePredict(self.ui.images_predict_page_import_list)

    def evnClearImagesButtonClickedPageResults(self):
        if self.PredictedImagesPixMap and showDialog('Clear all images', 'Are you sure?', QMessageBox.Information):
            self.ui.images_results_page_import_list.clear()
            self.PredictedImagesPixMap = {}
            imageLabelFrame(self.ui.label_results_page_selected_picture)
            self.ui.label_results_page_selected_picture.setText("Please predict images first.")

            buttons_tuple = [(self.ui.btn_results_page_clear_images, False),
                             (self.ui.btn_results_page_uncheck_all, False),
                             (self.ui.btn_results_page_check_all, False),
                             (self.ui.btn_results_page_delete_selected_images, False),
                             (self.ui.btn_results_page_save_images, False),
                             (self.ui.btn_results_page_save_csvs, False),
                             (self.ui.btn_results_page_save_images_and_csvs, False)]

            toggleButtonAndChangeStyle(buttons_tuple)
            self.updateNumOfImagesPageResults(self.ui.images_results_page_import_list)

    def evnClearImagesButtonClickedPageCalculation(self):
        if self.imagesForCalculationNpArray and showDialog('Clear all images', 'Are you sure?',
                                                           QMessageBox.Information):
            self.ui.images_calculation_page_import_list.clear()
            self.imagesForCalculationNpArray = {}
            self.imagesForCalculationPixMap = {}

            imageLabelFrame(self.ui.label_calculate_page_selected_picture)
            self.ui.label_calculate_page_selected_picture.setText("No results")

            buttons_tuple = [(self.ui.btn_calculation_page_clear_images, False),
                             (self.ui.btn_calculation_page_uncheck_all, False),
                             (self.ui.btn_calculation_page_check_all, False),
                             (self.ui.btn_calculation_page_delete_selected_images, False),
                             (self.ui.btn_calculation_page_save_images, False),
                             (self.ui.btn_calculation_page_save_csvs, False),
                             (self.ui.btn_calculation_page_send, False)
                             ]

            check_box_tuple = [
                (self.ui.frame_calculation_page_modifications_options_min_spin_box, False),
                (self.ui.frame_calculation_page_modifications_options_max_spin_box, False),
                (self.ui.frame_calculation_page_modifications_options_max_label, False),
                (self.ui.frame_calculation_page_modifications_options_min_label, False),
                (self.ui.check_box_show_and_calculate_centroid, False),
                (self.ui.check_box_show_external_contures, False),
                (self.ui.check_box_show_internal_contures, False)]

            toggleButtonAndChangeStyle(buttons_tuple)
            toggleCheckBoxAndChangeStyle(check_box_tuple)
            self.updateNumOfImagesPageCalculation(self.ui.images_calculation_page_import_list)

    def evnSendButtonClickedPageCalculation(self):

        checked_calculate_items = {}
        checked_min_max_values = {}

        import_list = self.ui.images_calculation_page_import_list
        show_external = self.ui.check_box_show_external_contures
        show_and_calculate_centroid = self.ui.check_box_show_and_calculate_centroid
        show_internal = self.ui.check_box_show_internal_contures

        check_box_flags = {
            'show_in_contours': show_internal.isChecked(),
            'show_ex_contours': show_external.isChecked(),
            'calc_centroid': show_and_calculate_centroid.isChecked()
        }

        for index in range(import_list.count()):
            if import_list.item(index).checkState() == 2:
                list_item = import_list.item(index)
                list_item_name = list_item.text()
                checked_calculate_items[list_item_name] = self.imagesForCalculationNpArray[list_item_name]
                checked_min_max_values[list_item_name] = (
                    self.imagesMinValues[list_item_name], self.imagesMaxValues[list_item_name])
        self.imagesDrawn, self.imagesAnalyse = analyze(checked_calculate_items, check_box_flags, checked_min_max_values,
                                                       num_of_bins=10)

    def evnCheckAllButtonClickedPagePredict(self):
        if showDialog('Check all images', 'Are you sure?', QMessageBox.Information):
            for index in range(self.ui.images_predict_page_import_list.count()):
                if self.ui.images_predict_page_import_list.item(index).checkState() == 0:
                    self.ui.images_predict_page_import_list.item(index).setCheckState(2)

            buttons_tuple = [(self.ui.btn_predict_page_check_all, False),
                             (self.ui.btn_predict_page_uncheck_all, True),
                             (self.ui.btn_predict_page_predict, True),
                             (self.ui.btn_predict_page_delete_selected_images, True)]

            toggleButtonAndChangeStyle(buttons_tuple)
            self.updateNumOfImagesPagePredict(self.ui.images_predict_page_import_list)

    def evnUncheckAllButtonClickedPagePredict(self):
        if showDialog('Uncheck all images', 'Are you sure?', QMessageBox.Information):
            for index in range(self.ui.images_predict_page_import_list.count()):
                if self.ui.images_predict_page_import_list.item(index).checkState() == 2:
                    self.ui.images_predict_page_import_list.item(index).setCheckState(0)

            buttons_tuple = [(self.ui.btn_predict_page_check_all, True),
                             (self.ui.btn_predict_page_uncheck_all, False),
                             (self.ui.btn_predict_page_predict, False),
                             (self.ui.btn_predict_page_delete_selected_images, False)]

            toggleButtonAndChangeStyle(buttons_tuple)
            self.updateNumOfImagesPagePredict(self.ui.images_predict_page_import_list)

    def evnCheckAllButtonClickedPageResults(self):
        if showDialog('Check all images', 'Are you sure?', QMessageBox.Information):
            for index in range(self.ui.images_results_page_import_list.count()):
                if self.ui.images_results_page_import_list.item(index).checkState() == 0:
                    self.ui.images_results_page_import_list.item(index).setCheckState(2)

            buttons_tuple = [(self.ui.btn_results_page_check_all, False),
                             (self.ui.btn_results_page_uncheck_all, True),
                             (self.ui.btn_results_page_delete_selected_images, True),
                             (self.ui.btn_results_page_custom_calculation, True),
                             (self.ui.btn_results_page_save_images, True),
                             (self.ui.btn_results_page_save_csvs, True),
                             (self.ui.btn_results_page_save_images_and_csvs, True),
                             ]

            toggleButtonAndChangeStyle(buttons_tuple)
            self.updateNumOfImagesPageResults(self.ui.images_results_page_import_list)

    def evnCheckAllButtonClickedPageCalculation(self):
        if showDialog('Check all images', 'Are you sure?', QMessageBox.Information):
            for index in range(self.ui.images_calculation_page_import_list.count()):
                if self.ui.images_calculation_page_import_list.item(index).checkState() == 0:
                    self.ui.images_calculation_page_import_list.item(index).setCheckState(2)

            buttons_tuple = [(self.ui.btn_calculation_page_check_all, False),
                             (self.ui.btn_calculation_page_uncheck_all, True),
                             (self.ui.btn_calculation_page_delete_selected_images, True),
                             (self.ui.btn_calculation_page_save_csvs, True),
                             (self.ui.btn_calculation_page_save_images, True),
                             (self.ui.btn_calculation_page_send, True),

                             ]

            check_box_tuple = [
                (self.ui.frame_calculation_page_modifications_options_min_spin_box, True),
                (self.ui.frame_calculation_page_modifications_options_max_spin_box, True),
                (self.ui.frame_calculation_page_modifications_options_max_label, True),
                (self.ui.frame_calculation_page_modifications_options_min_label, True),
                (self.ui.check_box_show_and_calculate_centroid, True),
                (self.ui.check_box_show_external_contures, True),
                (self.ui.check_box_show_internal_contures, True)]

            toggleButtonAndChangeStyle(buttons_tuple)
            toggleCheckBoxAndChangeStyle(check_box_tuple)
            self.updateNumOfImagesPageCalculation(self.ui.images_calculation_page_import_list)

    def evnUncheckAllButtonClickedPageResults(self):
        if showDialog('Uncheck all images', 'Are you sure?', QMessageBox.Information):
            for index in range(self.ui.images_results_page_import_list.count()):
                if self.ui.images_results_page_import_list.item(index).checkState() == 2:
                    self.ui.images_results_page_import_list.item(index).setCheckState(0)

            buttons_tuple = [(self.ui.btn_results_page_check_all, True),
                             (self.ui.btn_results_page_uncheck_all, False),
                             (self.ui.btn_results_page_delete_selected_images, False),
                             (self.ui.btn_results_page_custom_calculation, False),
                             (self.ui.btn_results_page_save_images, False),
                             (self.ui.btn_results_page_save_csvs, False),
                             (self.ui.btn_results_page_save_images_and_csvs, False)
                             ]

            toggleButtonAndChangeStyle(buttons_tuple)
            self.updateNumOfImagesPageResults(self.ui.images_results_page_import_list)

    def evnUncheckAllButtonClickedPageCalculation(self):
        if showDialog('Uncheck all images', 'Are you sure?', QMessageBox.Information):
            for index in range(self.ui.images_calculation_page_import_list.count()):
                if self.ui.images_calculation_page_import_list.item(index).checkState() == 2:
                    self.ui.images_calculation_page_import_list.item(index).setCheckState(0)

            buttons_tuple = [(self.ui.btn_calculation_page_check_all, True),
                             (self.ui.btn_calculation_page_uncheck_all, False),
                             (self.ui.btn_calculation_page_delete_selected_images, False),
                             (self.ui.btn_calculation_page_save_images, False),
                             (self.ui.btn_calculation_page_save_csvs, False),
                             (self.ui.btn_calculation_page_send, False),

                             ]

            check_box_tuple = [
                (self.ui.frame_calculation_page_modifications_options_min_spin_box, False),
                (self.ui.frame_calculation_page_modifications_options_max_spin_box, False),
                (self.ui.frame_calculation_page_modifications_options_max_label, False),
                (self.ui.frame_calculation_page_modifications_options_min_label, False),
                (self.ui.check_box_show_and_calculate_centroid, False),
                (self.ui.check_box_show_external_contures, False),
                (self.ui.check_box_show_internal_contures, False)]

            toggleButtonAndChangeStyle(buttons_tuple)
            toggleCheckBoxAndChangeStyle(check_box_tuple)
            self.updateNumOfImagesPageCalculation(self.ui.images_calculation_page_import_list)

    def updateNumOfImagesPagePredict(self, widget_list):
        self.ui.label_predict_page_images.setText(
            f"Images: {widget_list.count()} Checked: {numOfCheckedItems(widget_list)}")

    def updateNumOfImagesPageResults(self, widget_list):
        self.ui.label_results_page_images.setText(
            f"Images: {widget_list.count()} Checked: {numOfCheckedItems(widget_list)}")

    def updateNumOfImagesPageCalculation(self, widget_list):
        self.ui.label_calculate_page_images.setText(
            f"Images: {widget_list.count()} Checked: {numOfCheckedItems(widget_list)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
