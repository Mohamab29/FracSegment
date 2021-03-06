# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_graphs.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import UI.icons_rc


class Ui_Graphs(object):
    def setupUi(self, Graphs):
        Graphs.setObjectName("Graphs")
        Graphs.resize(830, 513)
        Graphs.setMinimumSize(QtCore.QSize(800, 500))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/assets/icons/graph-icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Graphs.setWindowIcon(icon)
        Graphs.setStyleSheet("background-color: #282c34")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(Graphs)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame_graph_info = QtWidgets.QFrame(Graphs)
        self.frame_graph_info.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_graph_info.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_graph_info.setObjectName("frame_graph_info")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_graph_info)
        self.verticalLayout.setContentsMargins(9, 9, 9, 0)
        self.verticalLayout.setSpacing(9)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_graph_2 = QtWidgets.QFrame(self.frame_graph_info)
        self.label_graph_2.setStyleSheet("#pagesContainer QPushButton {\n"
                                         "    border: 2px solid rgb(52, 59, 72);\n"
                                         "    border-radius: 5px;    \n"
                                         "    background-color: rgb(52, 59, 72);\n"
                                         "}\n"
                                         "#pagesContainer QPushButton:hover {\n"
                                         "    background-color: rgb(57, 65, 80);\n"
                                         "    border: 2px solid rgb(61, 70, 86);\n"
                                         "}\n"
                                         "#pagesContainer QPushButton:pressed {    \n"
                                         "    background-color: rgb(35, 40, 49);\n"
                                         "    border: 2px solid rgb(43, 50, 61);\n"
                                         "}")
        self.label_graph_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_graph_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_graph_2.setObjectName("label_graph_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.label_graph_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_graph = QtWidgets.QLabel(self.label_graph_2)
        self.label_graph.setScaledContents(True)
        self.label_graph.setAlignment(QtCore.Qt.AlignCenter)
        self.label_graph.setObjectName("label_graph")
        self.gridLayout_2.addWidget(self.label_graph, 0, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.verticalLayout.addWidget(self.label_graph_2)
        self.frame_widgets = QtWidgets.QFrame(self.frame_graph_info)
        self.frame_widgets.setMaximumSize(QtCore.QSize(800, 500))
        self.frame_widgets.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_widgets.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_widgets.setObjectName("frame_widgets")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_widgets)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 9)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.btn_save_graph = QtWidgets.QPushButton(self.frame_widgets)
        self.btn_save_graph.setMinimumSize(QtCore.QSize(150, 25))
        self.btn_save_graph.setStyleSheet(" QPushButton {\n"
                                          "    border: 2px solid rgb(52, 59, 72);\n"
                                          "    border-radius: 5px;    \n"
                                          "    background-color: rgb(52, 59, 72);\n"
                                          "    color: #fff\n"
                                          "}\n"
                                          " QPushButton:hover {\n"
                                          "    background-color: rgb(57, 65, 80);\n"
                                          "    border: 2px solid rgb(61, 70, 86);\n"
                                          "}\n"
                                          " QPushButton:pressed {    \n"
                                          "    background-color: rgb(35, 40, 49);\n"
                                          "    border: 2px solid rgb(43, 50, 61);\n"
                                          "}")
        self.btn_save_graph.setObjectName("btn_save_graph")
        self.horizontalLayout_2.addWidget(self.btn_save_graph)
        self.combobox_graphs = QtWidgets.QComboBox(self.frame_widgets)
        self.combobox_graphs.setMinimumSize(QtCore.QSize(150, 25))
        self.combobox_graphs.setStyleSheet("QComboBox{\n"
                                           "    background-color: rgb(27, 29, 35);\n"
                                           "    border-radius: 5px;\n"
                                           "    border: 2px solid rgb(33, 37, 43);\n"
                                           "    padding: 5px;\n"
                                           "    padding-left: 10px;\n"
                                           "    color: #fff\n"
                                           "}\n"
                                           "QComboBox:hover{\n"
                                           "    border: 2px solid rgb(64, 71, 88);\n"
                                           "}\n"
                                           "QComboBox::drop-down {\n"
                                           "    subcontrol-origin: padding;\n"
                                           "    subcontrol-position: top right;\n"
                                           "    width: 25px; \n"
                                           "    border-left-width: 3px;\n"
                                           "    border-left-color: rgba(39, 44, 54, 150);\n"
                                           "    border-left-style: solid;\n"
                                           "    border-top-right-radius: 3px;\n"
                                           "    border-bottom-right-radius: 3px;    \n"
                                           "    background-image: url(./assets/icons/cil-arrow-bottom.png);\n"
                                           "    background-position: center;\n"
                                           "    background-repeat: no-reperat;\n"
                                           " }\n"
                                           "QComboBox QAbstractItemView {\n"
                                           "    color: rgb(255, 121, 198);    \n"
                                           "    background-color: rgb(33, 37, 43);\n"
                                           "    padding: 10px;\n"
                                           "    selection-background-color: rgb(39, 44, 54);\n"
                                           "}")
        self.combobox_graphs.setObjectName("combobox_graphs")
        self.combobox_graphs.addItem("")
        self.combobox_graphs.addItem("")
        self.combobox_graphs.addItem("")
        self.combobox_graphs.addItem("")
        self.horizontalLayout_2.addWidget(self.combobox_graphs)
        self.verticalLayout.addWidget(self.frame_widgets, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout_2.addWidget(self.frame_graph_info)

        self.retranslateUi(Graphs)
        QtCore.QMetaObject.connectSlotsByName(Graphs)

    def retranslateUi(self, Graphs):
        _translate = QtCore.QCoreApplication.translate
        Graphs.setWindowTitle(_translate("Graphs", "Graphs"))
        self.label_graph.setText(_translate("Graphs",
                                            "<html><head/><body><p><span style=\" color:#ffffff;\">Please select Graph</span></p></body></html>"))
        self.btn_save_graph.setText(_translate("Graphs", "Save Selected Graph"))
        self.combobox_graphs.setItemText(0, _translate("Graphs", "Area"))
        self.combobox_graphs.setItemText(1, _translate("Graphs", "Ratio"))
        self.combobox_graphs.setItemText(2, _translate("Graphs", "Local Avg Depth"))
        self.combobox_graphs.setItemText(3, _translate("Graphs", "Global Avg Depth"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Graphs = QtWidgets.QWidget()
    ui = Ui_Graphs()
    ui.setupUi(Graphs)
    Graphs.show()
    sys.exit(app.exec_())
