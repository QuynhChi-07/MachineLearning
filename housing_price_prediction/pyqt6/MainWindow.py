# File: MainWindow.py (GIẢ ĐỊNH - CẦN THAY THẾ BẰNG FILE TỰ SINH)
# Tự động sinh ra sau khi chạy lệnh pyuic6 -x MainWindow.ui -o MainWindow.py
from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 850)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Layout Chính (Vertical)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")

        # Top Panel
        self.frame_top = QtWidgets.QFrame(self.centralwidget)
        self.frame_top.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame_top.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_top.setObjectName("frame_top")
        self.horizontalLayout_title = QtWidgets.QHBoxLayout(self.frame_top)
        self.label_title = QtWidgets.QLabel(self.frame_top)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        self.label_title.setFont(font)
        self.label_title.setText("House Pricing Prediction - Faculty of Information Systems")
        self.label_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.horizontalLayout_title.addWidget(self.label_title)
        self.verticalLayout.addWidget(self.frame_top)

        # Center Panel - Container cho các chức năng
        self.frame_center = QtWidgets.QFrame(self.centralwidget)
        self.frame_center.setObjectName("frame_center")
        self.verticalLayout_center = QtWidgets.QVBoxLayout(self.frame_center)

        # -- 1. Chọn Dataset & Train/Evaluate --
        self.gridLayout_config = QtWidgets.QGridLayout()

        # Hàng 1: Dataset
        self.gridLayout_config.addWidget(QtWidgets.QLabel("Select Dataset:"), 0, 0)
        self.lineEdit_dataset_path = QtWidgets.QLineEdit(self.frame_center)
        self.lineEdit_dataset_path.setObjectName("lineEdit_dataset_path")
        self.gridLayout_config.addWidget(self.lineEdit_dataset_path, 0, 1)
        self.btn_pick_dataset = QtWidgets.QPushButton("1. Pick Dataset", self.frame_center)
        self.btn_pick_dataset.setObjectName("btn_pick_dataset")
        self.gridLayout_config.addWidget(self.btn_pick_dataset, 0, 2)
        self.btn_view_dataset = QtWidgets.QPushButton("2. View Dataset", self.frame_center)
        self.btn_view_dataset.setObjectName("btn_view_dataset")
        self.gridLayout_config.addWidget(self.btn_view_dataset, 0, 3)

        # Hàng 2: Training
        self.gridLayout_config.addWidget(QtWidgets.QLabel("Training Rate (%):"), 1, 0)
        self.lineEdit_training_rate = QtWidgets.QLineEdit(self.frame_center)
        self.lineEdit_training_rate.setObjectName("lineEdit_training_rate")
        self.gridLayout_config.addWidget(self.lineEdit_training_rate, 1, 1)
        self.btn_train_model = QtWidgets.QPushButton("3. Train Model", self.frame_center)
        self.btn_train_model.setObjectName("btn_train_model")
        self.gridLayout_config.addWidget(self.btn_train_model, 1, 2)
        self.btn_evaluate_model = QtWidgets.QPushButton("4. Evaluate Model", self.frame_center)
        self.btn_evaluate_model.setObjectName("btn_evaluate_model")
        self.gridLayout_config.addWidget(self.btn_evaluate_model, 1, 3)
        self.label_train_status = QtWidgets.QLabel("", self.frame_center)
        self.label_train_status.setObjectName("label_train_status")
        self.gridLayout_config.addWidget(self.label_train_status, 1, 4)

        self.verticalLayout_center.addLayout(self.gridLayout_config)

        # -- 2. Evaluation Panel (Splitter) --
        self.splitter_evaluation = QtWidgets.QSplitter(self.frame_center)
        self.splitter_evaluation.setOrientation(QtCore.Qt.Orientation.Horizontal)

        # Bảng Kết quả
        self.tableWidget_evaluation = QtWidgets.QTableWidget(self.splitter_evaluation)
        self.tableWidget_evaluation.setObjectName("tableWidget_evaluation")
        self.tableWidget_evaluation.setColumnCount(7)
        self.tableWidget_evaluation.setHorizontalHeaderLabels(
            ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
             'Area Population', 'Original Price', 'Prediction Price'])

        # Panel Metrics
        self.frame_metrics = QtWidgets.QFrame(self.splitter_evaluation)
        self.frame_metrics.setObjectName("frame_metrics")
        self.verticalLayout_metrics = QtWidgets.QVBoxLayout(self.frame_metrics)

        self.verticalLayout_metrics.addWidget(QtWidgets.QLabel("Coefficient:"))
        self.textEdit_coefficient = QtWidgets.QTextEdit(self.frame_metrics)
        self.textEdit_coefficient.setObjectName("textEdit_coefficient")
        self.textEdit_coefficient.setMaximumHeight(200)
        self.verticalLayout_metrics.addWidget(self.textEdit_coefficient)

        self.gridLayout_metrics = QtWidgets.QGridLayout()

        self.gridLayout_metrics.addWidget(QtWidgets.QLabel("Mean Absolute Error(MAE):"), 0, 0)
        self.lineEdit_mae = QtWidgets.QLineEdit(self.frame_metrics)
        self.lineEdit_mae.setObjectName("lineEdit_mae")
        self.lineEdit_mae.setReadOnly(True)
        self.gridLayout_metrics.addWidget(self.lineEdit_mae, 0, 1)

        self.gridLayout_metrics.addWidget(QtWidgets.QLabel("Mean Square Error(MSE):"), 1, 0)
        self.lineEdit_mse = QtWidgets.QLineEdit(self.frame_metrics)
        self.lineEdit_mse.setObjectName("lineEdit_mse")
        self.lineEdit_mse.setReadOnly(True)
        self.gridLayout_metrics.addWidget(self.lineEdit_mse, 1, 1)

        self.gridLayout_metrics.addWidget(QtWidgets.QLabel("Root Mean Square Error(RMSE):"), 2, 0)
        self.lineEdit_rmse = QtWidgets.QLineEdit(self.frame_metrics)
        self.lineEdit_rmse.setObjectName("lineEdit_rmse")
        self.lineEdit_rmse.setReadOnly(True)
        self.gridLayout_metrics.addWidget(self.lineEdit_rmse, 2, 1)

        self.btn_save_model = QtWidgets.QPushButton("5. Save Model", self.frame_metrics)
        self.btn_save_model.setObjectName("btn_save_model")
        self.gridLayout_metrics.addWidget(self.btn_save_model, 3, 1)

        self.verticalLayout_metrics.addLayout(self.gridLayout_metrics)
        self.verticalLayout_center.addWidget(self.splitter_evaluation)

        # -- 3. Load Model & Prediction --
        self.gridLayout_prediction = QtWidgets.QGridLayout()

        # Load Model
        self.btn_load_model = QtWidgets.QPushButton("6. Load Model", self.frame_center)
        self.btn_load_model.setObjectName("btn_load_model")
        self.gridLayout_prediction.addWidget(self.btn_load_model, 0, 0)
        self.comboBox_models = QtWidgets.QComboBox(self.frame_center)
        self.comboBox_models.setObjectName("comboBox_models")
        self.gridLayout_prediction.addWidget(self.comboBox_models, 0, 1)

        # Input Fields
        self.gridLayout_prediction.addWidget(QtWidgets.QLabel("Avg. Area Income:"), 1, 0)
        self.lineEdit_income = QtWidgets.QLineEdit(self.frame_center)
        self.lineEdit_income.setObjectName("lineEdit_income")
        self.gridLayout_prediction.addWidget(self.lineEdit_income, 1, 1)

        self.gridLayout_prediction.addWidget(QtWidgets.QLabel("Avg. Area House Age:"), 2, 0)
        self.lineEdit_house_age = QtWidgets.QLineEdit(self.frame_center)
        self.lineEdit_house_age.setObjectName("lineEdit_house_age")
        self.gridLayout_prediction.addWidget(self.lineEdit_house_age, 2, 1)

        self.gridLayout_prediction.addWidget(QtWidgets.QLabel("Avg. Area Number of Rooms:"), 3, 0)
        self.lineEdit_num_rooms = QtWidgets.QLineEdit(self.frame_center)
        self.lineEdit_num_rooms.setObjectName("lineEdit_num_rooms")
        self.gridLayout_prediction.addWidget(self.lineEdit_num_rooms, 3, 1)

        self.gridLayout_prediction.addWidget(QtWidgets.QLabel("Avg. Area Number of Bedrooms:"), 4, 0)
        self.lineEdit_num_bedrooms = QtWidgets.QLineEdit(self.frame_center)
        self.lineEdit_num_bedrooms.setObjectName("lineEdit_num_bedrooms")
        self.gridLayout_prediction.addWidget(self.lineEdit_num_bedrooms, 4, 1)

        self.gridLayout_prediction.addWidget(QtWidgets.QLabel("Area Population:"), 5, 0)
        self.lineEdit_population = QtWidgets.QLineEdit(self.frame_center)
        self.lineEdit_population.setObjectName("lineEdit_population")
        self.gridLayout_prediction.addWidget(self.lineEdit_population, 5, 1)

        self.btn_predict = QtWidgets.QPushButton("7. Prediction House Pricing", self.frame_center)
        self.btn_predict.setObjectName("btn_predict")
        self.gridLayout_prediction.addWidget(self.btn_predict, 6, 1)

        self.gridLayout_prediction.addWidget(QtWidgets.QLabel("Prediction Price:"), 7, 0)
        self.lineEdit_prediction_price = QtWidgets.QLineEdit(self.frame_center)
        self.lineEdit_prediction_price.setObjectName("lineEdit_prediction_price")
        self.lineEdit_prediction_price.setReadOnly(True)
        self.gridLayout_prediction.addWidget(self.lineEdit_prediction_price, 7, 1)

        self.verticalLayout_center.addLayout(self.gridLayout_prediction)
        self.verticalLayout.addWidget(self.frame_center)

        MainWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)