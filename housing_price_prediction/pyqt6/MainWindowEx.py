# File: MainWindowEx.py
import os
import pickle
import glob
import datetime
import numpy as np
import pandas as pd
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Import lớp giao diện tự động sinh từ MainWindow.ui
# Giả sử bạn đã chạy: pyuic6 -x MainWindow.ui -o MainWindow.py
from MainWindow import Ui_MainWindow  # Đảm bảo file MainWindow.py có tồn tại

# --- Thiết lập đường dẫn tuyệt đối theo yêu cầu của bạn ---
# Đường dẫn DATA và MODEL cần được chỉnh sửa theo máy của bạn
# Trong code này, tôi dùng đường dẫn tương đối (từ thư mục hiện tại)
# để dễ chạy, nhưng bạn có thể thay thế bằng đường dẫn tuyệt đối đã cung cấp
DATA_PATH = "dataset/USA_Housing.csv"  # Đường dẫn tương đối
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
# Đảm bảo thư mục 'models' tồn tại nếu bạn muốn lưu model
os.makedirs(MODEL_DIR, exist_ok=True)
INITIAL_MODEL_PATH = "housingmodel.zip"  # Model mặc định


# --------------------------------------------------------------------------

class MainWindowEx(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindowEx, self).__init__(parent)
        self.setupUi(self)
        self.lm = None
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.initial_setup()
        self.connect_signals()
        self.load_initial_model()

    def initial_setup(self):
        # Thiết lập giá trị mặc định cho các widget
        self.lineEdit_dataset_path.setText(DATA_PATH)
        self.lineEdit_training_rate.setText("80")

        # Cập nhật danh sách model có sẵn
        self.update_model_list()

        # Cài đặt TreeWidget/TableWidget cho Evaluation
        # Nếu dùng QTableWidget (đơn giản hơn QTreeWidget)
        self.tableWidget_evaluation.setColumnCount(7)
        self.tableWidget_evaluation.setHorizontalHeaderLabels([
            'Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
            'Avg. Area Number of Bedrooms', 'Area Population', 'Original Price', 'Prediction Price'
        ])
        self.tableWidget_evaluation.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )

    def connect_signals(self):
        # Kết nối các nút với hàm xử lý
        self.btn_pick_dataset.clicked.connect(self.do_pick_data)
        self.btn_view_dataset.clicked.connect(self.do_view_dataset)
        self.btn_train_model.clicked.connect(self.do_train)
        self.btn_evaluate_model.clicked.connect(self.do_evaluation)
        self.btn_save_model.clicked.connect(self.do_save_model)
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_predict.clicked.connect(self.do_prediction)

    def load_initial_model(self):
        # Tương đương với logic trong __init__ của Tkinter
        try:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), INITIAL_MODEL_PATH)
            with open(model_path, 'rb') as f:
                self.lm = pickle.load(f)
            QMessageBox.information(self, "Thông báo", "Model mặc định đã được nạp thành công.")
        except FileNotFoundError:
            QMessageBox.warning(self, "Cảnh báo", "Không tìm thấy model mặc định. Sẽ sử dụng LinearRegression trống.")
            from sklearn.linear_model import LinearRegression
            self.lm = LinearRegression()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể load model mặc định: {e}")
            from sklearn.linear_model import LinearRegression
            self.lm = LinearRegression()

    def update_model_list(self):
        self.comboBox_models.clear()

        # Tìm các file model housingmodel_*.zip hoặc housingmodel_*.pkl trong thư mục models
        model_options_zip = glob.glob(os.path.join(MODEL_DIR, "housingmodel_*.zip"))
        model_options_pkl = glob.glob(os.path.join(MODEL_DIR, "housingmodel_*.pkl"))

        all_model_paths = model_options_zip + model_options_pkl
        self.model_options = [os.path.basename(f) for f in all_model_paths]

        if not self.model_options:
            self.model_options = ["(Chưa có model nào)"]

        self.comboBox_models.addItems(self.model_options)

    # --- Các hàm chức năng chính ---

    def do_pick_data(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Chọn Dataset", "", "Dataset CSV (*.csv);;All Files (*)")
        if fileName:
            self.lineEdit_dataset_path.setText(fileName)

    def do_view_dataset(self):
        file_path = self.lineEdit_dataset_path.text()
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Cảnh báo", "File dataset không tồn tại!")
            return

        try:
            df = pd.read_csv(file_path)

            # Hiển thị dữ liệu trong một cửa sổ mới (đơn giản hóa)
            # Trong thực tế, bạn nên tạo một QDialog mới cho DataSetViewer
            # Để đơn giản, tôi chỉ hiển thị 5 dòng đầu tiên trong message box
            QMessageBox.information(self, "Dữ liệu Dataset", f"5 dòng đầu tiên:\n{df.head().to_string()}")

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi đọc file dataset: {e}")

    def do_train(self):
        file_path = self.lineEdit_dataset_path.text()
        try:
            ratio = float(self.lineEdit_training_rate.text()) / 100
            if not (0.0 < ratio <= 1.0):
                raise ValueError("Tỉ lệ Training Rate phải nằm trong khoảng (0, 100].")

            self.df = pd.read_csv(file_path)
            self.X = self.df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                              'Avg. Area Number of Bedrooms', 'Area Population']]
            self.y = self.df['Price']

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=1 - ratio, random_state=101
            )

            self.lm = LinearRegression()
            self.lm.fit(self.X_train, self.y_train)

            self.label_train_status.setText("Trained is finished")
            QMessageBox.information(self, "Thông báo", "Trained is finished")

        except ValueError as ve:
            QMessageBox.warning(self, "Lỗi Input", str(ve))
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi huấn luyện mô hình: {e}")

    def do_evaluation(self):
        if self.lm is None:
            QMessageBox.showwarning(self, "Cảnh báo", "Vui lòng huấn luyện mô hình trước!")
            return
        if self.X_test is None:
            QMessageBox.showwarning(self, "Cảnh báo", "Vui lòng Train Model trước để có dữ liệu test!")
            return

        try:
            # 1. Hiển thị Coefficient
            self.textEdit_coefficient.clear()
            coeff_df = pd.DataFrame(self.lm.coef_, self.X.columns, columns=['Coefficient'])
            self.textEdit_coefficient.setText(f"Intercept: {self.lm.intercept_}\n\n{coeff_df.to_string()}")

            # 2. Dự đoán và Hiển thị trong TableWidget
            predictions = self.lm.predict(self.X_test)

            self.tableWidget_evaluation.setRowCount(len(self.X_test))

            for i in range(len(self.X_test)):
                # Dữ liệu đầu vào
                input_data = self.X_test.iloc[i].tolist()
                # Giá gốc
                original_price = self.y_test.iloc[i]
                # Giá dự đoán
                predicted_price = predictions[i]

                row_values = input_data + [original_price, predicted_price]

                for j, value in enumerate(row_values):
                    # Format float cho dễ nhìn
                    formatted_value = f"{value:,.2f}" if isinstance(value, (int, float)) else str(value)
                    item = QTableWidgetItem(formatted_value)
                    self.tableWidget_evaluation.setItem(i, j, item)

            # 3. Tính và Hiển thị Metric
            mae = metrics.mean_absolute_error(self.y_test, predictions)
            mse = metrics.mean_squared_error(self.y_test, predictions)
            rmse = np.sqrt(mse)

            self.lineEdit_mae.setText(f"{mae:,.2f}")
            self.lineEdit_mse.setText(f"{mse:,.2f}")
            self.lineEdit_rmse.setText(f"{rmse:,.2f}")

            self.label_train_status.setText("Evaluation is finished")
            QMessageBox.information(self, "Thông báo", "Evaluation is finished")

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi đánh giá mô hình: {e}")

    def do_save_model(self):
        if self.lm is None:
            QMessageBox.showwarning(self, "Cảnh báo", "Chưa có mô hình nào để lưu!")
            return

        reply = QMessageBox.question(self, "Xác nhận", "Bạn có chắc muốn lưu mô hình hiện tại?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            return

        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Nên dùng .pkl để lưu model Python
            model_name = f"housingmodel_{timestamp}.pkl"
            model_path = os.path.join(MODEL_DIR, model_name)

            with open(model_path, "wb") as f:
                pickle.dump(self.lm, f)

            QMessageBox.information(self, "Thành công", f"Đã lưu mô hình: {model_name}")
            self.update_model_list()

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể lưu model:\n{e}")

    def load_model(self):
        model_name = self.comboBox_models.currentText()

        if model_name == "(Chưa có model nào)":
            QMessageBox.showwarning(self, "Cảnh báo", "Chưa có mô hình nào để nạp.")
            return

        try:
            model_path = os.path.join(MODEL_DIR, model_name)

            # Nếu model mặc định ban đầu là .zip, kiểm tra trong thư mục gốc của app
            if model_name.endswith('.zip') and not os.path.exists(model_path):
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)

            with open(model_path, "rb") as file:
                self.lm = pickle.load(file)
            QMessageBox.information(self, "Thành công", f"Đã nạp mô hình: {model_name}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể load model:\n{e}")

    def do_prediction(self):
        if self.lm is None:
            QMessageBox.showwarning(self, "Cảnh báo", "Vui lòng nạp hoặc huấn luyện mô hình trước!")
            return

        try:
            income = float(self.lineEdit_income.text())
            house_age = float(self.lineEdit_house_age.text())
            num_rooms = float(self.lineEdit_num_rooms.text())
            num_bedrooms = float(self.lineEdit_num_bedrooms.text())
            population = float(self.lineEdit_population.text())

            input_data = np.array([[income, house_age, num_rooms, num_bedrooms, population]])

            # Kiểm tra nếu mô hình được huấn luyện trên DataFrame, nó cần tên cột
            # Tuy nhiên, LinearRegression của sklearn thường nhận mảng numpy
            result = self.lm.predict(input_data)

            self.lineEdit_prediction_price.setText(f"{result[0]:,.2f}")

        except ValueError:
            QMessageBox.warning(self, "Lỗi Input", "Vui lòng nhập số hợp lệ vào tất cả các trường.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi trong quá trình dự đoán: {e}")