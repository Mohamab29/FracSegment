import os
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QDesktopWidget, QFrame
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from ui_main import Ui_MainWindow
from PIL.ImageQt import ImageQt
from BackEnd.segmenting import segment
from BackEnd.analyze_dimples import analyze, saveImagesAnalysisToCSV, find_max_area
import cv2
import io
from PIL import Image
from PyQt5.QtCore import QBuffer
from styles import widgets_style


def toggleWidgetAndChangeStyle(pair):
    """
    Change mode (enable/disable) and button design (darker/lither).

    :param pair: a tuple of widget and mode to change to (True/False) For example: (predict_btn, False)
    """
    for p in pair:
        widget = p[0]
        term = p[1]
        widget_class_name = widget.__class__.__name__
        widget.setEnabled(term)
        widget.setStyleSheet(widgets_style[widget_class_name][str(term)])


def setListItemItemStyle(item):
    """
    Change mode (enable/disable) and button design (darker/lither).

    :param item: a QListWidgetItem of one of the lists in the app.
    """
    font = QtGui.QFont()
    font.setBold(True)
    font.setWeight(75)
    item.setFont(font)


def showDialog(title, message, icon):
    """
    Shows Dialog when we need it.

    :param title: a Dialog title.
    :param message: a Dialog message.
    :param icon: a Dialog icon.
    """
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


def countCheckedItems(widget_list, what_to_count):
    """
    Calculation of the amount of records marked in the list.

    :param widget_list: the current list for which the number of marked records is counted.
    :param what_to_count: A parameter that tells the function what to count, for example: number of checked items.
    :returns: boolean or number (of checked items).
    """

    def isAtLeastOneItemChecked():
        for index in range(widget_list.count()):
            if widget_list.item(index).checkState() == 2:
                return True
        return False

    def isAtLeastOneItemNotChecked():
        for index in range(widget_list.count()):
            if widget_list.item(index).checkState() == 0:
                return True
        return False

    def isNoItemChecked():
        for index in range(widget_list.count()):
            if widget_list.item(index).checkState() == 2:
                return False
        return True

    def numOfCheckedItems():
        counter = 0
        for index in range(widget_list.count()):
            if widget_list.item(index).checkState() == 2:
                counter = counter + 1
        return counter

    return {'at_list_one_checked': isAtLeastOneItemChecked,
            'at_list_one_not_checked': isAtLeastOneItemNotChecked,
            'not_items_checked': isNoItemChecked,
            'num_of_check_items': numOfCheckedItems
            }[what_to_count]()


def imageLabelFrame(label, frame_shape=0, frame_shadow=0, line_width=0):
    """
    When an PixMap image is displayed on the screen, this function adds a frame to it.

    :param label: The current label on which we put the picture.
    :param frame_shape: Frame shape style.
    :param frame_shadow: Frame shadow style.
    :param line_width: Frame line width.
    """
    label.setFrameShape(frame_shape)
    label.setFrameShadow(frame_shadow)
    label.setLineWidth(line_width)


def evnImageListItemDoubleClicked(item, dic, page):
    """
    The event is triggered when the user double-clicks on an image on one of the pages in the app.

    :param item: The current QListWidgetItem that we double click it.
    :param dic: Frame shape style.
    :param page: In which image the image was clicked.
    """

    def pagePredict():
        if item:
            image = Image.open(dic[item.text()], 'r')
            image.show()

    def pageResultsAndCalculation():
        if item:
            buffer = QBuffer()
            buffer.open(QBuffer.ReadWrite)
            dic[item.text()].save(buffer, item.text().split('.')[-1])
            pil_im = Image.open(io.BytesIO(buffer.data()))
            pil_im.show()

    {
        'predict': pagePredict,
        'results': pageResultsAndCalculation,
        'calculation': pageResultsAndCalculation
    }[page]()


def convertCvImage2QtImage(img, mode):
    """
    Converting an image to a pix map. from a numpy array to and image object to a Pixmap.

    :returns: a Pixmap object
    """
    if mode == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return QtGui.QPixmap.fromImage(ImageQt(Image.fromarray(img).convert(mode)))


def convertCvImage2QtImageRGB(img, mode):
    """
    Converting an image to a pix map. from a numpy array to and image object to a Pixmap.

    :returns: a Pixmap object
    """
    if mode == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = QtGui.QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QtGui.QImage.Format_RGB888)
    pix = QtGui.QPixmap(image.copy())
    return pix


def updateNumOfImages(widget_list, label):
    """
    Updates the number of complete photos in the list and the number of checked photos.
    """
    label.setText(
        f"Images: {widget_list.count()} Checked: {countCheckedItems(widget_list, 'num_of_check_items')}")


def addImageNameToList(image_name, list):
    list_item = QtWidgets.QListWidgetItem()
    list_item.setCheckState(QtCore.Qt.Checked)
    list_item.setText(image_name)
    setListItemItemStyle(list_item)
    list.addItem(list_item)


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
        """
        Centralizes the app window.
        """
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def setActions(self):
        """
        This is where all the connections between the events and the buttons in the app are made
        """

        # toggle event listener
        self.ui.btn_toggle.clicked.connect(self.evnBtnToggleClicked)

        # pages event listeners
        self.ui.btn_page_predict.clicked.connect(lambda: self.evnPageClicked(self.ui.frame_predict_page))
        self.ui.btn_page_results.clicked.connect(lambda: self.evnPageClicked(self.ui.frame_results_page))
        self.ui.btn_page_calculation.clicked.connect(lambda: self.evnPageClicked(self.ui.frame_calculation_page))
        self.ui.btn_page_help.clicked.connect(lambda: self.evnPageClicked(self.ui.frame_help_page))

        # predict page event listeners
        self.ui.btn_predict_page_load_images.clicked.connect(self.evnLoadImagesButtonClicked)
        self.ui.btn_predict_page_clear_images.clicked.connect(self.evnClearImagesButtonClickedPagePredict)
        self.ui.btn_predict_page_check_all.clicked.connect(self.evnCheckAllButtonClickedPagePredict)
        self.ui.btn_predict_page_uncheck_all.clicked.connect(self.evnUncheckAllButtonClickedPagePredict)
        self.ui.btn_predict_page_delete_selected_images.clicked.connect(
            lambda: self.evnDeleteSelectedImagesButton(self.ui.images_predict_page_import_list,
                                                       self.ui.label_predict_page_selected_picture,
                                                       [self.imageListPathDict],
                                                       "predict",
                                                       [(self.ui.btn_predict_page_clear_images, False)]))
        self.ui.btn_predict_page_predict.clicked.connect(self.evnPredictButtonClicked)
        self.ui.images_predict_page_import_list.itemClicked.connect(self.evnImageListItemClickedPagePredict)
        self.ui.images_predict_page_import_list.itemDoubleClicked.connect(
            lambda item: evnImageListItemDoubleClicked(item, self.imageListPathDict, "predict"))
        self.ui.images_predict_page_import_list.currentItemChanged.connect(
            lambda item: self.evnCurrentItemChanged(item, self.ui.label_predict_page_selected_picture,
                                                    self.imageListPathDict, 'predict'))

        # results page event listeners
        self.ui.btn_results_page_clear_images.clicked.connect(self.evnClearImagesButtonClickedPageResults)
        self.ui.btn_results_page_check_all.clicked.connect(self.evnCheckAllButtonClickedPageResults)
        self.ui.btn_results_page_uncheck_all.clicked.connect(self.evnUncheckAllButtonClickedPageResults)
        self.ui.btn_results_page_delete_selected_images.clicked.connect(
            lambda: self.evnDeleteSelectedImagesButton(self.ui.images_results_page_import_list,
                                                       self.ui.label_results_page_selected_picture,
                                                       [self.PredictedImagesPixMap],
                                                       "results",
                                                       [(self.ui.btn_results_page_clear_images, False)]))

        self.ui.btn_results_page_save_images.clicked.connect(self.evnSaveSelectedImagesButtonClickedPageResults)
        self.ui.btn_results_page_custom_calculation.clicked.connect(self.evnCustomCalculationButtonClicked)
        self.ui.btn_results_page_save_csvs.clicked.connect(self.evnSaveCsvsButtonClickedPageResults)
        self.ui.btn_results_page_save_images_and_csvs.clicked.connect(self.evnSaveImageAndCsvsButtonClickedPageResults)
        self.ui.images_results_page_import_list.itemClicked.connect(self.evnImageListItemClickedPageResults)
        self.ui.images_results_page_import_list.itemDoubleClicked.connect(
            lambda item: evnImageListItemDoubleClicked(item, self.PredictedImagesPixMap, "results"))
        self.ui.images_results_page_import_list.currentItemChanged.connect(
            lambda item: self.evnCurrentItemChanged(item, self.ui.label_results_page_selected_picture,
                                                    self.PredictedImagesPixMap, 'results'))

        # calculation page event listeners
        self.ui.btn_calculation_page_clear_images.clicked.connect(self.evnClearImagesButtonClickedPageCalculation)
        self.ui.btn_calculation_page_send.clicked.connect(self.evnSendButtonClickedPageCalculation)
        self.ui.btn_calculation_page_check_all.clicked.connect(self.evnCheckAllButtonClickedPageCalculation)
        self.ui.btn_calculation_page_uncheck_all.clicked.connect(self.evnUncheckAllButtonClickedPageCalculation)
        self.ui.btn_calculation_page_delete_selected_images.clicked.connect(
            lambda: self.evnDeleteSelectedImagesButton(self.ui.images_calculation_page_import_list,
                                                       self.ui.label_calculate_page_selected_picture,
                                                       [self.imagesForCalculationNpArray,
                                                        self.imagesForCalculationPixMap,
                                                        self.imagesDrawn,
                                                        self.imagesAnalyse],
                                                       "calculation",
                                                       [(self.ui.btn_results_page_clear_images, False)]))
        self.ui.btn_calculation_page_save_images.clicked.connect(self.evnSaveImagesButtonClickedPageCalculation)
        self.ui.btn_calculation_page_save_csvs.clicked.connect(self.evnSaveCsvsButtonClickedPageCalculation)
        self.ui.images_calculation_page_import_list.itemClicked.connect(
            lambda item: self.evnImageListItemClickedPageCalculation(item,
                                                                     self.ui.label_calculate_page_selected_picture))
        self.ui.images_calculation_page_import_list.itemDoubleClicked.connect(
            lambda item: evnImageListItemDoubleClicked(item, self.imagesForCalculationPixMap, "calculation"))
        self.ui.images_calculation_page_import_list.currentItemChanged.connect(
            lambda item: self.evnCurrentItemChanged(item, self.ui.label_calculate_page_selected_picture,
                                                    self.imagesDrawn, "calculation"))
        self.ui.frame_calculation_page_modifications_options_max_spin_box.valueChanged.connect(
            lambda: self.evnChangeMaxOrMinValuePageCalculation(self.imagesMaxValues,
                                                               self.ui.frame_calculation_page_modifications_options_max_spin_box))
        self.ui.frame_calculation_page_modifications_options_min_spin_box.valueChanged.connect(
            lambda: self.evnChangeMaxOrMinValuePageCalculation(self.imagesMinValues,
                                                               self.ui.frame_calculation_page_modifications_options_min_spin_box))

    def evnPredictButtonClicked(self):
        """
        This event runs when you press the predict button on the main page of the app.
        """
        import_list = self.ui.images_predict_page_import_list
        checked_items = []
        allredy_predicted_images = []

        for index in range(import_list.count()):
            real_image_name = import_list.item(index).text()
            split_image_name = real_image_name.split('/')[-1].split('.')
            predicted_image_name = f"{split_image_name[0]}_predicted.{split_image_name[1]}"
            if import_list.item(index).checkState() == 2:
                if predicted_image_name not in self.PredictedImagesPixMap.keys():
                    checked_items.append(self.imageListPathDict[real_image_name])
                else:
                    allredy_predicted_images.append(real_image_name)

        if len(allredy_predicted_images) == 1:
            showDialog('Already predict image', f'The image: {allredy_predicted_images[0]} already predicted',
                       QMessageBox.Warning)
        if len(allredy_predicted_images) > 1:
            names_of_predicted_images = ''
            for name in allredy_predicted_images:
                names_of_predicted_images = names_of_predicted_images + ' ' + name
            showDialog('Already predict images', f'The images: {names_of_predicted_images} already predicted',
                       QMessageBox.Warning)

        num_of_image_to_predict = countCheckedItems(import_list, "num_of_check_items") - len(allredy_predicted_images)
        if num_of_image_to_predict > 0:
            if showDialog('Predict images',
                          f'Predict for {num_of_image_to_predict} images?',
                          QMessageBox.Question):

                segmented_images = segment(checked_items)
                segmented_images_size = segmented_images.__len__()

                if segmented_images_size:
                    for index in range(segmented_images_size):
                        segmented_image = segmented_images[index]
                        split_image_name = checked_items[index].split('/')[-1].split('.')
                        image_name = f"{split_image_name[0]}_predicted.{split_image_name[1]}"
                        self.PredictedImagesPixMap[image_name] = convertCvImage2QtImage(segmented_image, "L")
                        self.PredictedImagesNpArray[image_name] = segmented_image
                        addImageNameToList(image_name, self.ui.images_results_page_import_list)

                    widgets_tuples = [
                        (self.ui.btn_results_page_clear_images, True),
                        (self.ui.btn_results_page_uncheck_all, True),
                        (self.ui.btn_results_page_delete_selected_images, True),
                        (self.ui.btn_results_page_save_images, True),
                        (self.ui.btn_results_page_custom_calculation, True),
                        (self.ui.btn_results_page_save_csvs, True),
                        (self.ui.btn_results_page_save_images_and_csvs, True)
                    ]

                    toggleWidgetAndChangeStyle(widgets_tuples)
                    import_list = self.ui.images_results_page_import_list
                    selected_result_list_size = len(import_list.selectedItems())

                    if not selected_result_list_size:
                        label = self.ui.label_results_page_selected_picture
                        label.setPixmap(
                            QtGui.QPixmap(convertCvImage2QtImage(segmented_images[len(segmented_images) - 1], "L")))
                        imageLabelFrame(label, QFrame.StyledPanel, QFrame.Sunken, 3)

                    updateNumOfImages(import_list, self.ui.label_results_page_images)

                    if showDialog('Prediction Succeeded', 'Move to the results page?', QMessageBox.Question):
                        self.ui.stackedWidget.setCurrentWidget(self.ui.frame_results_page)

    def evnCustomCalculationButtonClicked(self):
        """
        This event runs when you press the custom calculation button on the second page (results) of the app.
        """
        results_page_list = self.ui.images_results_page_import_list
        if showDialog('Calculate images',
                      f'Calculate for {countCheckedItems(results_page_list, "num_of_check_items")} images?',
                      QMessageBox.Question):

            checked_min_max_values = {}

            for index in range(results_page_list.count()):
                list_item = results_page_list.item(index)
                list_item_name = list_item.text()
                list_item_calc_name = list_item_name.replace('_predicted', '_calculated')
                if results_page_list.item(index).checkState() == 2:
                    if list_item_calc_name not in self.imagesForCalculationNpArray.keys():
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
                    addImageNameToList(name, self.ui.images_calculation_page_import_list)
                    names.append(name)

                widgets_tuples = [
                    (self.ui.btn_calculation_page_save_images, True),
                    (self.ui.btn_calculation_page_save_csvs, True),
                    (self.ui.btn_calculation_page_delete_selected_images, True),
                    (self.ui.btn_calculation_page_clear_images, True),
                    (self.ui.btn_calculation_page_send, True),
                    (self.ui.btn_calculation_page_uncheck_all, True),
                    (self.ui.frame_calculation_page_modifications_options_min_spin_box, True),
                    (self.ui.frame_calculation_page_modifications_options_max_spin_box, True),
                    (self.ui.frame_calculation_page_modifications_options_max_label, True),
                    (self.ui.frame_calculation_page_modifications_options_min_label, True),
                    (self.ui.check_box_show_and_calculate_centroid, True),
                    (self.ui.check_box_show_external_contures, True),
                    (self.ui.check_box_show_internal_contures, True)
                ]

                toggleWidgetAndChangeStyle(widgets_tuples)
                import_list = self.ui.images_calculation_page_import_list
                selected_calculation_list_size = len(import_list.selectedItems())

                if not selected_calculation_list_size:
                    last_image = self.imagesDrawn[names[-1]]
                    label = self.ui.label_calculate_page_selected_picture
                    label.setPixmap(QtGui.QPixmap(convertCvImage2QtImageRGB(last_image.copy(), "RGB")))
                    imageLabelFrame(label, QFrame.StyledPanel, QFrame.Sunken, 3)
                updateNumOfImages(import_list, self.ui.label_calculate_page_images)

                if showDialog('Calculation Succeeded', 'Move to the calculation page?', QMessageBox.Question):
                    self.ui.stackedWidget.setCurrentWidget(self.ui.frame_calculation_page)

    def evnSaveCsvsButtonClickedPageResults(self):
        """
        This event runs when we press the save scvs button on the second page (results) of the app.
        """
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

    def evnSaveSelectedImagesButtonClickedPageResults(self):
        """
        This event runs when we press the save selected image button on the second page (results) of the app.
        """
        checked_items = {}

        for index in range(self.ui.images_results_page_import_list.count()):
            if self.ui.images_results_page_import_list.item(index).checkState() == 2:
                list_item = self.ui.images_results_page_import_list.item(index)
                checked_items[list_item.text()] = self.PredictedImagesPixMap[list_item.text()]

        path = QFileDialog.getExistingDirectory(self, "Choose Folder")

        if path and checked_items:
            if not os.path.exists(f"{path}/files/predicted_images/"):
                os.makedirs(f"{path}/files/predicted_images/")

            for image_name in checked_items:
                pixmap_image = checked_items[image_name]
                image_path = f"{path}/files/predicted_images/{image_name}"
                image_type = image_name.split('.')[-1]
                pixmap_image.save(image_path, image_type)

    def evnSaveImageAndCsvsButtonClickedPageResults(self):
        """
        This event runs when we press the save images and csvs button on the second page (results) of the app.
        """
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

    def evnCurrentItemChanged(self, item, label, dict, page):
        """
        This event runs whenever there is a change in a particular list. When the user clicks on an entry on
        a particular page, the image will appear in the middle of the screen.
        """

        def predict_results_pages():
            if item:
                label.setPixmap(QtGui.QPixmap(dict[item.text()]))
                imageLabelFrame(label, QFrame.StyledPanel, QFrame.Sunken, 3)

        def calculation_page():
            if item:
                image = convertCvImage2QtImageRGB(dict[item.text()], "RGB")
                label.setPixmap(image)
                imageLabelFrame(label, QFrame.StyledPanel, QFrame.Sunken, 3)
                self.currentItemClickedNameCalcPage = item.text()

        {'predict': predict_results_pages,
         'results': predict_results_pages,
         'calculation': calculation_page
         }[page]()

    def evnChangeMaxOrMinValuePageCalculation(self, dict, spin_box):
        """
        This event saves the minimum or maximum value of the size of the dimples in pixels.
        """
        dict[self.currentItemClickedNameCalcPage] = spin_box.value()
        spin_box.setValue(dict[self.currentItemClickedNameCalcPage])

    def sharedTermsPagePredict(self):
        """
        Turns on and off buttons according to the number of records marked on the main page (predict).
        We also use use this method on the evnDeleteSelectedImagesButton event.
        """
        widget_list = self.ui.images_predict_page_import_list
        updateNumOfImages(widget_list, self.ui.label_predict_page_images)

        if not countCheckedItems(widget_list, 'at_list_one_checked'):
            toggleWidgetAndChangeStyle([(self.ui.btn_predict_page_uncheck_all, False)])
        else:
            toggleWidgetAndChangeStyle([(self.ui.btn_predict_page_uncheck_all, True)])
        if countCheckedItems(widget_list, 'at_list_one_not_checked'):
            toggleWidgetAndChangeStyle([(self.ui.btn_predict_page_check_all, True)])
        else:
            toggleWidgetAndChangeStyle([(self.ui.btn_predict_page_check_all, False)])
        if countCheckedItems(widget_list, 'num_of_check_items'):
            toggleWidgetAndChangeStyle([(self.ui.btn_predict_page_predict, True)])
            toggleWidgetAndChangeStyle([(self.ui.btn_predict_page_delete_selected_images, True)])
        else:
            toggleWidgetAndChangeStyle([(self.ui.btn_predict_page_predict, False)])
            toggleWidgetAndChangeStyle([(self.ui.btn_predict_page_delete_selected_images, False)])

    def sharedTermsPageResults(self):
        """
        Turns on and off buttons according to the number of records marked on the seconds page (results).
        We also use use this method on the evnDeleteSelectedImagesButton event.
        """
        widget_list = self.ui.images_results_page_import_list
        updateNumOfImages(widget_list, self.ui.label_results_page_images)

        if not countCheckedItems(widget_list, 'at_list_one_checked'):
            toggleWidgetAndChangeStyle([(self.ui.btn_results_page_uncheck_all, False)])
        else:
            toggleWidgetAndChangeStyle([(self.ui.btn_results_page_uncheck_all, True)])
        if countCheckedItems(widget_list, 'at_list_one_not_checked'):
            toggleWidgetAndChangeStyle([(self.ui.btn_results_page_check_all, True)])
        else:
            toggleWidgetAndChangeStyle([(self.ui.btn_results_page_check_all, False)])
        if countCheckedItems(widget_list, 'num_of_check_items'):
            toggleWidgetAndChangeStyle([(self.ui.btn_results_page_delete_selected_images, True)])
        else:
            toggleWidgetAndChangeStyle([(self.ui.btn_results_page_delete_selected_images, False)])
        if not countCheckedItems(widget_list, 'num_of_check_items'):
            widgets_tuples = [(self.ui.btn_results_page_custom_calculation, False),
                              (self.ui.btn_results_page_save_images, False),
                              (self.ui.btn_results_page_save_csvs, False),
                              (self.ui.btn_results_page_save_images_and_csvs, False),
                              ]
            toggleWidgetAndChangeStyle(widgets_tuples)
        else:
            widgets_tuples = [(self.ui.btn_results_page_custom_calculation, True),
                              (self.ui.btn_results_page_save_images, True),
                              (self.ui.btn_results_page_save_csvs, True),
                              (self.ui.btn_results_page_save_images_and_csvs, True),
                              ]
            toggleWidgetAndChangeStyle(widgets_tuples)

    def sharedTermsPageCalculation(self):
        """
        Turns on and off buttons according to the number of records marked on the thired page (calculation).
        We also use use this method on the evnDeleteSelectedImagesButton event.
        """
        widget_list = self.ui.images_calculation_page_import_list
        updateNumOfImages(widget_list, self.ui.label_calculate_page_images)

        if not countCheckedItems(widget_list, 'at_list_one_checked'):
            toggleWidgetAndChangeStyle([(self.ui.btn_calculation_page_uncheck_all, False)])
        else:
            toggleWidgetAndChangeStyle([(self.ui.btn_calculation_page_uncheck_all, True)])
        if countCheckedItems(widget_list, 'at_list_one_not_checked'):
            toggleWidgetAndChangeStyle([(self.ui.btn_calculation_page_check_all, True)])
        else:
            toggleWidgetAndChangeStyle([(self.ui.btn_calculation_page_check_all, False)])
        if countCheckedItems(widget_list, 'num_of_check_items'):
            toggleWidgetAndChangeStyle([(self.ui.btn_calculation_page_delete_selected_images, True)])
        else:
            toggleWidgetAndChangeStyle([(self.ui.btn_calculation_page_delete_selected_images, False)])
        if not countCheckedItems(widget_list, 'num_of_check_items'):
            widgets_tuples = [(self.ui.btn_calculation_page_clear_images, False),
                              (self.ui.btn_calculation_page_save_images, False),
                              (self.ui.btn_calculation_page_save_csvs, False),
                              (self.ui.btn_calculation_page_send, False),
                              (self.ui.frame_calculation_page_modifications_options_min_spin_box, False),
                              (self.ui.frame_calculation_page_modifications_options_max_spin_box, False),
                              (self.ui.frame_calculation_page_modifications_options_max_label, False),
                              (self.ui.frame_calculation_page_modifications_options_min_label, False),
                              (self.ui.check_box_show_and_calculate_centroid, False),
                              (self.ui.check_box_show_external_contures, False),
                              (self.ui.check_box_show_internal_contures, False)
                              ]
            toggleWidgetAndChangeStyle(widgets_tuples)
        else:
            widgets_tuples = [(self.ui.btn_calculation_page_clear_images, True),
                              (self.ui.btn_calculation_page_save_images, True),
                              (self.ui.btn_calculation_page_save_csvs, True),
                              (self.ui.btn_calculation_page_send, True),
                              (self.ui.frame_calculation_page_modifications_options_min_spin_box, True),
                              (self.ui.frame_calculation_page_modifications_options_max_spin_box, True),
                              (self.ui.frame_calculation_page_modifications_options_max_label, True),
                              (self.ui.frame_calculation_page_modifications_options_min_label, True),
                              (self.ui.check_box_show_and_calculate_centroid, True),
                              (self.ui.check_box_show_external_contures, True),
                              (self.ui.check_box_show_internal_contures, True)
                              ]
            toggleWidgetAndChangeStyle(widgets_tuples)

    def evnImageListItemClickedPagePredict(self):
        """
        This event listening to image click on the main page (predict).
        """
        self.sharedTermsPagePredict()

    def evnImageListItemClickedPageResults(self):
        """
        This event listening to image click on the second page (results).
        """
        self.sharedTermsPageResults()

    def evnImageListItemClickedPageCalculation(self, item, label):
        """
        This event listening to image click on the third page (calculation).
        :param item: The current QListWidgetItem that the user clicking on.
        :param label: The QLabel that changed to QPixmap image.
        """
        if item:
            self.ui.frame_calculation_page_modifications_options_max_spin_box.setValue(
                self.imagesMaxValues[item.text()])
            self.ui.frame_calculation_page_modifications_options_min_spin_box.setValue(
                self.imagesMinValues[item.text()])
            image = convertCvImage2QtImageRGB(self.imagesDrawn[item.text()], "RGB")
            label.setPixmap(image)
            imageLabelFrame(label, QFrame.StyledPanel, QFrame.Sunken, 3)
            self.currentItemClickedNameCalcPage = item.text()
        self.sharedTermsPageCalculation()

    def evnPageClicked(self, page_frame):
        """
        The event is activated when you click on navigation on the various pages.
        :param page_frame: The page the user wants to get.
        """
        self.ui.stackedWidget.setCurrentWidget(page_frame)

    def evnBtnToggleClicked(self):
        """
        The event triggers the toggleMenu method.
        """
        self.toggleMenu(150, True)

    def toggleMenu(self, max_width, enable):
        """
        Activates the toggle button and adds or removes the names of the pages.
        :param max_width: The maximum width to which the page menu will open.
        :param enable: A Boolean variable that announces whether the page menu is open or closed.
        """
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
        """
        Upload photos to the main page of the app
        """
        file_to_open = "Image Files (*.png *.jpg *.bmp *.tiff *.tif)"
        res = QFileDialog.getOpenFileNames(None, "Open File", "/", file_to_open)

        if len(res[0]) > 0:
            self.renderInputPictureList(res[0])

    def renderInputPictureList(self, pictures_input_path):
        """
        Renders the name list of the images on the main page and adds the
        image names of the images and their paths to the dictionary named imageListPathDict,
        When the key of each entry in the dictionary is the image name And the value is the path of the image.

        :param pictures_input_path: List of the images path.
        """
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
            widgets_tuples = [(self.ui.btn_predict_page_predict, True),
                              (self.ui.btn_predict_page_clear_images, True),
                              (self.ui.btn_predict_page_uncheck_all, True),
                              (self.ui.btn_predict_page_delete_selected_images, True)]
            toggleWidgetAndChangeStyle(widgets_tuples)

        label = self.ui.label_predict_page_selected_picture
        last_picture_name = pictures_input_path[len(pictures_input_path) - 1].split('/')[-1]
        label.setPixmap(QtGui.QPixmap(self.imageListPathDict[last_picture_name]))
        imageLabelFrame(label, QFrame.StyledPanel, QFrame.Sunken, 3)

        updateNumOfImages(self.ui.images_predict_page_import_list, self.ui.label_predict_page_images)

    def evnDeleteSelectedImagesButton(self, list, label, dict_list, page_name, buttons_to_toggle):
        """
        Deletes the checked images from the dictionaries that hold them on the various pages.

        :param list:
        :param label:
        :param dict_list:
        :param page_name:
        :param buttons_to_toggle:
        """
        checked_items = []

        if showDialog('Delete the selected images', 'Are you sure?', QMessageBox.Information):
            for index in range(list.count()):
                if list.item(index).checkState() == 2:
                    list_item = list.item(index)
                    checked_items.append(list_item)

            for item in checked_items:
                list.takeItem(list.row(item))
                for dict in dict_list:
                    dict.pop(item.text())

            {
                'predict': self.sharedTermsPagePredict,
                'results': self.sharedTermsPageResults,
                'calculation': self.sharedTermsPageCalculation
            }[page_name]()

            if not len(list):
                label.setText("Please load and select image.")
                imageLabelFrame(label, 0, 0, 0)
                toggleWidgetAndChangeStyle(buttons_to_toggle)

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
            self.imageListPathDict.clear()
            imageLabelFrame(self.ui.label_predict_page_selected_picture, 0, 0, 0)
            self.ui.label_predict_page_selected_picture.setText("Please load and select image.")

            widgets_tuples = [(self.ui.btn_predict_page_uncheck_all, False),
                              (self.ui.btn_predict_page_check_all, False),
                              (self.ui.btn_predict_page_predict, False),
                              (self.ui.btn_predict_page_delete_selected_images, False),
                              (self.ui.btn_predict_page_clear_images, False)]

            toggleWidgetAndChangeStyle(widgets_tuples)
            updateNumOfImages(self.ui.images_predict_page_import_list,
                              self.ui.label_predict_page_images)

    def evnClearImagesButtonClickedPageResults(self):
        if self.PredictedImagesPixMap and showDialog('Clear all images', 'Are you sure?', QMessageBox.Information):
            self.ui.images_results_page_import_list.clear()
            self.PredictedImagesPixMap.clear()

            imageLabelFrame(self.ui.label_results_page_selected_picture)
            self.ui.label_results_page_selected_picture.setText("Please predict images first.")

            widgets_tuples = [(self.ui.btn_results_page_clear_images, False),
                              (self.ui.btn_results_page_uncheck_all, False),
                              (self.ui.btn_results_page_check_all, False),
                              (self.ui.btn_results_page_delete_selected_images, False),
                              (self.ui.btn_results_page_save_images, False),
                              (self.ui.btn_results_page_save_csvs, False),
                              (self.ui.btn_results_page_save_images_and_csvs, False)]

            toggleWidgetAndChangeStyle(widgets_tuples)
            updateNumOfImages(self.ui.images_results_page_import_list, self.ui.label_results_page_images)

    def evnClearImagesButtonClickedPageCalculation(self):
        if self.imagesForCalculationNpArray and showDialog('Clear all images', 'Are you sure?',
                                                           QMessageBox.Information):
            self.ui.images_calculation_page_import_list.clear()
            self.imagesForCalculationNpArray.clear()
            self.imagesForCalculationPixMap.clear()
            self.imagesDrawn.clear()
            self.imagesAnalyse.clear()
            imageLabelFrame(self.ui.label_calculate_page_selected_picture)
            self.ui.label_calculate_page_selected_picture.setText("No results")
            widgets_tuples = [(self.ui.btn_calculation_page_clear_images, False),
                              (self.ui.btn_calculation_page_uncheck_all, False),
                              (self.ui.btn_calculation_page_check_all, False),
                              (self.ui.btn_calculation_page_delete_selected_images, False),
                              (self.ui.btn_calculation_page_save_images, False),
                              (self.ui.btn_calculation_page_save_csvs, False),
                              (self.ui.btn_calculation_page_send, False),
                              (self.ui.frame_calculation_page_modifications_options_min_spin_box, False),
                              (self.ui.frame_calculation_page_modifications_options_max_spin_box, False),
                              (self.ui.frame_calculation_page_modifications_options_max_label, False),
                              (self.ui.frame_calculation_page_modifications_options_min_label, False),
                              (self.ui.check_box_show_and_calculate_centroid, False),
                              (self.ui.check_box_show_external_contures, False),
                              (self.ui.check_box_show_internal_contures, False)
                              ]
            toggleWidgetAndChangeStyle(widgets_tuples)
            updateNumOfImages(self.ui.images_calculation_page_import_list, self.ui.label_calculate_page_images)

    def evnSendButtonClickedPageCalculation(self):
        import_list = self.ui.images_calculation_page_import_list
        if showDialog('Send custom properties',
                      f'Send for {countCheckedItems(import_list, "num_of_check_items")} images?',
                      QMessageBox.Question):
            checked_calculate_items = {}
            checked_min_max_values = {}
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
            self.imagesDrawn, self.imagesAnalyse = analyze(checked_calculate_items, check_box_flags,
                                                           checked_min_max_values,
                                                           num_of_bins=10)

    def evnCheckAllButtonClickedPagePredict(self):
        if showDialog('Check all images', 'Are you sure?', QMessageBox.Information):
            for index in range(self.ui.images_predict_page_import_list.count()):
                if self.ui.images_predict_page_import_list.item(index).checkState() == 0:
                    self.ui.images_predict_page_import_list.item(index).setCheckState(2)

            widgets_tuples = [(self.ui.btn_predict_page_check_all, False),
                              (self.ui.btn_predict_page_uncheck_all, True),
                              (self.ui.btn_predict_page_predict, True),
                              (self.ui.btn_predict_page_delete_selected_images, True)]
            toggleWidgetAndChangeStyle(widgets_tuples)
            updateNumOfImages(self.ui.images_predict_page_import_list,
                              self.ui.label_predict_page_images)

    def evnUncheckAllButtonClickedPagePredict(self):
        if showDialog('Uncheck all images', 'Are you sure?', QMessageBox.Information):
            for index in range(self.ui.images_predict_page_import_list.count()):
                if self.ui.images_predict_page_import_list.item(index).checkState() == 2:
                    self.ui.images_predict_page_import_list.item(index).setCheckState(0)

            widgets_tuples = [(self.ui.btn_predict_page_check_all, True),
                              (self.ui.btn_predict_page_uncheck_all, False),
                              (self.ui.btn_predict_page_predict, False),
                              (self.ui.btn_predict_page_delete_selected_images, False)]

            toggleWidgetAndChangeStyle(widgets_tuples)
            updateNumOfImages(self.ui.images_predict_page_import_list,
                              self.ui.label_predict_page_images)

    def evnCheckAllButtonClickedPageResults(self):
        if showDialog('Check all images', 'Are you sure?', QMessageBox.Information):
            for index in range(self.ui.images_results_page_import_list.count()):
                if self.ui.images_results_page_import_list.item(index).checkState() == 0:
                    self.ui.images_results_page_import_list.item(index).setCheckState(2)

            widgets_tuples = [(self.ui.btn_results_page_check_all, False),
                              (self.ui.btn_results_page_uncheck_all, True),
                              (self.ui.btn_results_page_delete_selected_images, True),
                              (self.ui.btn_results_page_custom_calculation, True),
                              (self.ui.btn_results_page_save_images, True),
                              (self.ui.btn_results_page_save_csvs, True),
                              (self.ui.btn_results_page_save_images_and_csvs, True),
                              ]

            toggleWidgetAndChangeStyle(widgets_tuples)
            updateNumOfImages(self.ui.images_results_page_import_list, self.ui.label_results_page_images)

    def evnCheckAllButtonClickedPageCalculation(self):
        if showDialog('Check all images', 'Are you sure?', QMessageBox.Information):
            for index in range(self.ui.images_calculation_page_import_list.count()):
                if self.ui.images_calculation_page_import_list.item(index).checkState() == 0:
                    self.ui.images_calculation_page_import_list.item(index).setCheckState(2)

            widgets_tuples = [(self.ui.btn_calculation_page_check_all, False),
                              (self.ui.btn_calculation_page_uncheck_all, True),
                              (self.ui.btn_calculation_page_delete_selected_images, True),
                              (self.ui.btn_calculation_page_save_csvs, True),
                              (self.ui.btn_calculation_page_save_images, True),
                              (self.ui.btn_calculation_page_send, True),
                              (self.ui.btn_calculation_page_clear_images, True),
                              (self.ui.frame_calculation_page_modifications_options_min_spin_box, True),
                              (self.ui.frame_calculation_page_modifications_options_max_spin_box, True),
                              (self.ui.frame_calculation_page_modifications_options_max_label, True),
                              (self.ui.frame_calculation_page_modifications_options_min_label, True),
                              (self.ui.check_box_show_and_calculate_centroid, True),
                              (self.ui.check_box_show_external_contures, True),
                              (self.ui.check_box_show_internal_contures, True)
                              ]

            toggleWidgetAndChangeStyle(widgets_tuples)
            updateNumOfImages(self.ui.images_calculation_page_import_list, self.ui.label_calculate_page_images)

    def evnUncheckAllButtonClickedPageResults(self):
        if showDialog('Uncheck all images', 'Are you sure?', QMessageBox.Information):
            for index in range(self.ui.images_results_page_import_list.count()):
                if self.ui.images_results_page_import_list.item(index).checkState() == 2:
                    self.ui.images_results_page_import_list.item(index).setCheckState(0)

            widgets_tuples = [(self.ui.btn_results_page_check_all, True),
                              (self.ui.btn_results_page_uncheck_all, False),
                              (self.ui.btn_results_page_delete_selected_images, False),
                              (self.ui.btn_results_page_custom_calculation, False),
                              (self.ui.btn_results_page_save_images, False),
                              (self.ui.btn_results_page_save_csvs, False),
                              (self.ui.btn_results_page_save_images_and_csvs, False)
                              ]

            toggleWidgetAndChangeStyle(widgets_tuples)
            updateNumOfImages(self.ui.images_results_page_import_list, self.ui.label_results_page_images)

    def evnUncheckAllButtonClickedPageCalculation(self):
        if showDialog('Uncheck all images', 'Are you sure?', QMessageBox.Information):
            for index in range(self.ui.images_calculation_page_import_list.count()):
                if self.ui.images_calculation_page_import_list.item(index).checkState() == 2:
                    self.ui.images_calculation_page_import_list.item(index).setCheckState(0)

            widgets_tuples = [(self.ui.btn_calculation_page_check_all, True),
                              (self.ui.btn_calculation_page_uncheck_all, False),
                              (self.ui.btn_calculation_page_delete_selected_images, False),
                              (self.ui.btn_calculation_page_save_images, False),
                              (self.ui.btn_calculation_page_save_csvs, False),
                              (self.ui.btn_calculation_page_send, False),
                              (self.ui.btn_calculation_page_clear_images, False),
                              (self.ui.frame_calculation_page_modifications_options_min_spin_box, False),
                              (self.ui.frame_calculation_page_modifications_options_max_spin_box, False),
                              (self.ui.frame_calculation_page_modifications_options_max_label, False),
                              (self.ui.frame_calculation_page_modifications_options_min_label, False),
                              (self.ui.check_box_show_and_calculate_centroid, False),
                              (self.ui.check_box_show_external_contures, False),
                              (self.ui.check_box_show_internal_contures, False)
                              ]

            toggleWidgetAndChangeStyle(widgets_tuples)
            updateNumOfImages(self.ui.images_calculation_page_import_list, self.ui.label_calculate_page_images)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
