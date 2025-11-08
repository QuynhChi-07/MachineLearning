from PyQt6.QtWidgets import QMessageBox, QTableWidgetItem
from PyQt6.QtCore import Qt

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np
import traceback

from on_KTGK.Connectors.Connector import Connector
from on_KTGK.UI.MainWindow import Ui_MainWindow


class MainWindowEx(Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.db = Connector()
        self.connected = False
        self.df_customers = None
        self.df_cat = None
        self.df_prod = None

    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)

        # Gán sự kiện nút (Giữ nguyên)
        self.pushButtonImportDB.clicked.connect(self.handleConnectDB)
        self.pushButtonTotalSaleByProduct.clicked.connect(self.queryTotalSaleByProduct)
        self.pushButtonTotalRevenueByCategory.clicked.connect(self.queryTotalRevenueByCategory)
        self.pushButtonRevenueByMonthYear.clicked.connect(self.queryRevenueByMonthYear)
        self.pushButtonFastDeliveryOrders.clicked.connect(self.queryFastDeliveryOrders)

        self.pushButtonGetCustomerDetails.clicked.connect(self.getCustomerDetails)
        self.pushButtonGetCustomerOrders.clicked.connect(self.getCustomerOrders)

        self.pushButtonTrainKMeans.clicked.connect(self.trainKMeans)
        self.pushButtonGetCustomersByCluster.clicked.connect(self.getCustomersByCluster)
        self.pushButtonPredictTrend.clicked.connect(self.predictTrend)

        # Thiết lập mặc định
        self.labelDBStatus.setText("Trạng thái: NGẮT KẾT NỐI")

    # ========================== 1. DB CONNECT ==========================
    def handleConnectDB(self):
        try:
            if not self.connected:
                conn = self.db.connect()
                if conn:
                    self.connected = True
                    self.labelDBStatus.setText("✅ Trạng thái: ĐÃ KẾT NỐI")
                    QMessageBox.information(None, "Kết nối", "Đã kết nối cơ sở dữ liệu thành công!")

                    # GỌI HÀM TẢI DỮ LIỆU VÀ ĐIỀN COMBOBOX
                    self.loadPredictionData()

                else:
                    raise Exception("Không thể kết nối tới cơ sở dữ liệu.")
            else:
                self.db.disConnect()
                self.connected = False
                self.labelDBStatus.setText("🔌 Trạng thái: NGẮT KẾT NỐI")
                QMessageBox.information(None, "Ngắt kết nối", "Đã ngắt kết nối cơ sở dữ liệu.")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(None, "Lỗi kết nối", f"Lỗi xảy ra khi kết nối DB: {str(e)}")

    # ========================== HÀM CHUNG THỰC THI SQL ==========================
    def executeQuery(self, sql, params=None):
        try:
            if not self.connected:
                QMessageBox.warning(None, "Lỗi", "Chưa kết nối cơ sở dữ liệu.")
                return

            df = pd.read_sql(sql, self.db.conn, params=params)
            self.showDataFrame(df)
            return df
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(None, "Lỗi SQL", f"Không thể thực thi câu lệnh SQL. Lỗi: {str(e)}")
            return pd.DataFrame()

    # ========================== 2-5. QUERY SQL (ĐÃ SỬA LỖI NGÀY THÁNG) ==========================
    def queryTotalSaleByProduct(self):
        sql = """
            SELECT p.Name AS ProductName, SUM(od.OrderQty) AS TotalQuantity
            FROM orderdetails od
            JOIN product p ON od.ProductID = p.ProductID
            GROUP BY p.Name
            ORDER BY TotalQuantity DESC;
        """
        self.executeQuery(sql)

    def queryTotalRevenueByCategory(self):
        sql = """
            SELECT c.CategoryID, c.Name AS CategoryName, SUM(od.OrderQty * od.UnitPrice) AS TotalRevenue
            FROM orderdetails od
            JOIN product p ON od.ProductID = p.ProductID
            JOIN subcategory sc ON p.ProductSubcategoryID = sc.SubcategoryID
            JOIN category c ON sc.CategoryID = c.CategoryID
            GROUP BY c.CategoryID, c.Name
            ORDER BY TotalRevenue DESC;
        """
        self.executeQuery(sql)

    # CHỨC NĂNG 4: SỬA LỖI NGÀY THÁNG VÀ LỌC NULL
    def queryRevenueByMonthYear(self):
        sql = """
            SELECT c.Name AS CategoryName,
                   YEAR(STR_TO_DATE(o.OrderDate, '%d/%m/%Y')) AS Year,
                   MONTH(STR_TO_DATE(o.OrderDate, '%d/%m/%Y')) AS Month,
                   SUM(od.OrderQty * od.UnitPrice) AS Revenue
            FROM orderdetails od
            JOIN orders o ON od.OrderID = o.OrderID
            JOIN product p ON od.ProductID = p.ProductID
            JOIN subcategory sc ON p.ProductSubcategoryID = sc.SubcategoryID
            JOIN category c ON sc.CategoryID = c.CategoryID
            WHERE STR_TO_DATE(o.OrderDate, '%d/%m/%Y') IS NOT NULL
            GROUP BY c.Name, 
                     YEAR(STR_TO_DATE(o.OrderDate, '%d/%m/%Y')),
                     MONTH(STR_TO_DATE(o.OrderDate, '%d/%m/%Y'))
            ORDER BY Year, Month, CategoryName;
        """
        self.executeQuery(sql)

    # CHỨC NĂNG 5: SỬA LỖI NGÀY THÁNG VÀ LỌC NULL
    def queryFastDeliveryOrders(self):
        sql = """
            SELECT o.OrderID, o.CustomerID, 
                   DATEDIFF(STR_TO_DATE(o.ShipDate, '%d/%m/%Y'), 
                            STR_TO_DATE(o.OrderDate, '%d/%m/%Y')) AS DeliveryDays
            FROM orders o
            WHERE STR_TO_DATE(o.OrderDate, '%d/%m/%Y') IS NOT NULL 
              AND STR_TO_DATE(o.ShipDate, '%d/%m/%Y') IS NOT NULL
              AND DATEDIFF(STR_TO_DATE(o.ShipDate, '%d/%m/%Y'), 
                           STR_TO_DATE(o.OrderDate, '%d/%m/%Y')) <= 3; 
        """
        self.executeQuery(sql)

    # ========================== 6-7. CUSTOMER FUNCTION (Giữ nguyên) ==========================
    def getCustomerDetails(self):
        customer_id = self.lineEditCustomerID.text().strip()
        if not customer_id:
            QMessageBox.warning(None, "Thiếu thông tin", "Vui lòng nhập CustomerID.")
            return

        sql = "SELECT * FROM customer WHERE CustomerID = %s;"
        self.executeQuery(sql, params=[customer_id])

    def getCustomerOrders(self):
        customer_id = self.lineEditCustomerID.text().strip()
        if not customer_id:
            QMessageBox.warning(None, "Thiếu thông tin", "Vui lòng nhập CustomerID.")
            return

        sql = """
            SELECT o.OrderID, o.OrderDate, o.ShipDate, SUM(od.OrderQty * od.UnitPrice) AS TotalAmount
            FROM orders o
            JOIN orderdetails od ON o.OrderID = od.OrderID
            WHERE o.CustomerID = %s
            GROUP BY o.OrderID, o.OrderDate, o.ShipDate;
        """
        self.executeQuery(sql, params=[customer_id])

    # ========================== 8. KMEANS CLUSTERING (ĐÃ TỐI ƯU HÓA) ==========================
    def trainKMeans(self):
        try:
            if not self.connected:
                QMessageBox.warning(None, "Lỗi", "Chưa kết nối cơ sở dữ liệu.")
                return

            k_text = self.lineEditNumberOfClusters.text().strip()
            if not k_text.isdigit():
                QMessageBox.warning(None, "Lỗi", "Số lượng cụm (K) phải là số nguyên dương.")
                return

            k = int(k_text)

            sql = """
                SELECT CustomerID, 
                       SUM(od.OrderQty) AS TotalQuantity,                   
                       SUM(od.OrderQty * od.UnitPrice) AS TotalSpend        
                FROM orderdetails od                                        
                JOIN orders o USING(OrderID) 
                GROUP BY CustomerID
                HAVING SUM(od.OrderQty) > 0; -- Tối ưu: Chỉ lấy khách hàng có đơn hàng
            """

            df = pd.read_sql(sql, self.db.conn)
            if df.empty:
                QMessageBox.warning(None, "Lỗi", "Không có dữ liệu khách hàng để huấn luyện.")
                return

            df['TotalQuantity'] = df['TotalQuantity'].fillna(0).astype(float)
            df['TotalSpend'] = df['TotalSpend'].fillna(0).astype(float)

            X = df[['TotalQuantity', 'TotalSpend']]

            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            df['Cluster'] = kmeans.fit_predict(X)

            self.df_customers = df
            self.showDataFrame(df)  # Hiển thị kết quả phân cụm

            # Định dạng lại trung tâm cụm cho dễ đọc
            center_df = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
            center_df.insert(0, 'Cluster ID', range(k))

            # Định dạng số cho trung tâm cụm
            center_df['TotalQuantity'] = center_df['TotalQuantity'].apply(lambda x: f'{x:,.2f}')
            center_df['TotalSpend'] = center_df['TotalSpend'].apply(lambda x: f'{x:,.2f}')

            self.textEditMLOutput.setText(
                f"Mô hình KMeans huấn luyện xong với K={k}\n\n"
                f"Trung tâm cụm:\n{center_df.to_string(index=False)}"
            )
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(None, "Lỗi ML", f"Không thể huấn luyện mô hình! Lỗi: {str(e)}")

    def getCustomersByCluster(self):
        if self.df_customers is None:
            QMessageBox.warning(None, "Lỗi", "Chưa huấn luyện mô hình KMeans.")
            return

        cluster_id_text = self.lineEditClusterID.text().strip()
        if not cluster_id_text.isdigit():
            QMessageBox.warning(None, "Lỗi", "Cluster ID phải là số.")
            return

        cluster_id = int(cluster_id_text)

        df = self.df_customers[self.df_customers['Cluster'] == cluster_id]

        if df.empty:
            QMessageBox.information(None, "Kết quả", f"Không tìm thấy khách hàng thuộc Cụm {cluster_id}.")
            self.showDataFrame(df)  # Vẫn gọi để clear bảng
            return

        # SỬA LỖI: Reset index của DataFrame đã lọc
        # Điều này đảm bảo index bắt đầu từ 0 và liền kề, giúp bảng Qt hiển thị đúng
        df = df.reset_index(drop=True)

        self.showDataFrame(df)

    # CHỨC NĂNG 9: SỬA LỖI NGÀY THÁNG VÀ ĐỊNH DẠNG HIỂN THỊ
    def predictTrend(self):
        try:
            if not self.connected:
                QMessageBox.warning(None, "Lỗi", "Chưa kết nối cơ sở dữ liệu.")
                return

            sql = """
                SELECT YEAR(STR_TO_DATE(o.OrderDate, '%d/%m/%Y')) AS Year, 
                       SUM(od.OrderQty * od.UnitPrice) AS Revenue
                FROM orders o
                JOIN orderdetails od ON o.OrderID = od.OrderID
                WHERE STR_TO_DATE(o.OrderDate, '%d/%m/%Y') IS NOT NULL
                GROUP BY YEAR(STR_TO_DATE(o.OrderDate, '%d/%m/%Y'))
                ORDER BY Year;
            """
            df = pd.read_sql(sql, self.db.conn)

            if len(df) < 2:
                QMessageBox.warning(None, "Lỗi Dự báo",
                                    "Không đủ dữ liệu (ít nhất 2 năm) để dự báo xu hướng. Dữ liệu hiện tại chỉ có từ một năm trở xuống.")
                self.textEditMLOutput.setText(
                    f"Dữ liệu theo năm:\n{df.to_string(index=False)}"
                )
                return

            df['Year'] = df['Year'].astype(float)
            df['Revenue'] = df['Revenue'].fillna(0).astype(float)

            X = df[["Year"]]
            y = df["Revenue"]

            model = LinearRegression()
            model.fit(X, y)

            next_year = int(df["Year"].max()) + 1
            next_year_array = np.array([[next_year]])
            pred = model.predict(next_year_array)[0]

            # Định dạng lại dữ liệu huấn luyện cho đẹp
            df_formatted = df.copy()
            df_formatted['Revenue'] = df_formatted['Revenue'].apply(lambda x: f'{x:,.2f}')

            self.textEditMLOutput.setText(
                f"🔮 Dự báo xu hướng doanh thu năm {next_year}: {pred:,.2f} USD\n\n"
                f"Dữ liệu huấn luyện:\n{df_formatted.to_string(index=False)}\n\n"
                f"Mô hình Linear Regression:\n"
                f"  - Hệ số góc (Slope): {model.coef_[0]:.2f}\n"
                f"  - Hệ số chặn (Intercept): {model.intercept_:,.2f}"
            )
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(None, "Lỗi Dự báo", f"Không thể dự báo xu hướng! Lỗi: {str(e)}")

    # ========================== 10. LOAD DATA CHO PREDICTION COMBOBOXES (Giữ nguyên) ==========================
    def loadPredictionData(self):
        try:
            if not self.connected:
                return

            # Tải danh mục và sản phẩm
            sql_cat = "SELECT CategoryID, Name FROM category ORDER BY Name;"
            df_cat = pd.read_sql(sql_cat, self.db.conn)
            sql_prod = "SELECT ProductID, Name FROM product ORDER BY Name;"
            df_prod = pd.read_sql(sql_prod, self.db.conn)

            # Lưu tạm DataFrame
            self.df_cat = df_cat
            self.df_prod = df_prod

            # ĐIỀN DỮ LIỆU VÀO COMBOBOXES
            # 1. Category
            self.comboBoxPredictionCategory.clear()
            self.comboBoxPredictionCategory.addItem("Chọn tất cả")
            self.comboBoxPredictionCategory.addItems(df_cat['Name'].tolist())

            # 2. Product
            self.comboBoxPredictionProduct.clear()
            self.comboBoxPredictionProduct.addItem("Chọn tất cả")
            self.comboBoxPredictionProduct.addItems(df_prod['Name'].tolist())

            # Ngắt kết nối sự kiện cũ
            try:
                self.comboBoxPredictionCategory.currentIndexChanged.disconnect()
            except TypeError:
                pass

            # Gán sự kiện mới
            self.comboBoxPredictionCategory.currentIndexChanged.connect(self.filterProductsByCat)

        except AttributeError:
            print("Lỗi UI: Các thành phần UI (ComboBox) chưa được khởi tạo. Bỏ qua tải dữ liệu Dropdown.")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(None, "Lỗi tải dữ liệu", f"Lỗi khi tải dữ liệu dropdown: {str(e)}")

    def filterProductsByCat(self):
        try:
            selected_cat_name = self.comboBoxPredictionCategory.currentText()
            self.comboBoxPredictionProduct.clear()
            self.comboBoxPredictionProduct.addItem("Chọn tất cả")

            if selected_cat_name == "Chọn tất cả":
                self.comboBoxPredictionProduct.addItems(self.df_prod['Name'].tolist())
            else:
                # Tìm CategoryID
                cat_id = self.df_cat[self.df_cat['Name'] == selected_cat_name]['CategoryID'].iloc[0]

                # Lọc sản phẩm theo CategoryID (Cần JOIN subcategory)
                sql_filter = f"""
                    SELECT p.Name 
                    FROM product p
                    JOIN subcategory sc ON p.ProductSubcategoryID = sc.SubcategoryID
                    WHERE sc.CategoryID = {cat_id}
                    ORDER BY p.Name;
                """
                df_filtered_prod = pd.read_sql(sql_filter, self.db.conn)
                self.comboBoxPredictionProduct.addItems(df_filtered_prod['Name'].tolist())
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(None, "Lỗi Lọc", f"Lỗi khi lọc sản phẩm theo danh mục: {str(e)}")

    # ========================== HIỂN THỊ DATAFRAME (ĐÃ CHỈNH SỬA) ==========================
    def showDataFrame(self, df: pd.DataFrame):
        self.tableWidgetResults.clear()

        if df.empty:
            QMessageBox.information(None, "Kết quả", "Không có dữ liệu.")
            self.tableWidgetResults.setRowCount(0)
            self.tableWidgetResults.setColumnCount(0)
            return

        self.tableWidgetResults.setRowCount(len(df))
        self.tableWidgetResults.setColumnCount(len(df.columns))
        self.tableWidgetResults.setHorizontalHeaderLabels(df.columns)

        for i, row in df.iterrows():
            for j, col in enumerate(df.columns):
                value = row[col]
                item = QTableWidgetItem(str(value))

                # Logic định dạng
                try:
                    # 1. Nếu là cột số tiền/doanh thu (giữ nguyên định dạng tiền tệ)
                    if ('Revenue' in col or 'Spend' in col or 'Amount' in col or 'Quantity' in col):
                        item = QTableWidgetItem(f"{value:,.2f}")
                        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

                    # 2. Nếu là cột ID, Year, Cluster (chuyển float sang int, bỏ .0)
                    elif ('ID' in col or 'Year' in col or 'Cluster' in col):
                        if isinstance(value, float) and value.is_integer():
                            # Chuyển 29994.0 thành 29994
                            item = QTableWidgetItem(str(int(value)))
                        else:
                            item = QTableWidgetItem(str(value))

                        item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

                    # 3. Các cột khác
                    else:
                        item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

                except:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

                self.tableWidgetResults.setItem(i, j, item)

        self.tableWidgetResults.resizeColumnsToContents()