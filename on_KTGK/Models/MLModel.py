import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np


class MLModel:
    def __init__(self, connector):
        self.conn = connector

    # (8) Phân cụm hành vi khách hàng bằng KMeans
    def customer_segmentation(self, n_clusters=3):
        sql = """
        SELECT c.CustomerID, SUM(od.OrderQty * od.UnitPrice) AS TotalSpend, 
               COUNT(DISTINCT o.OrderID) AS NumOrders
        FROM customer c                                                    
        JOIN orders o ON c.CustomerID = o.CustomerID
        JOIN orderdetails od ON o.OrderID = od.OrderID
        GROUP BY c.CustomerID;
        """
        df = self.conn.queryDataset(sql)

        # Bổ sung xử lý ép kiểu
        df["TotalSpend"] = df["TotalSpend"].fillna(0).astype(float)
        df["NumOrders"] = df["NumOrders"].fillna(0).astype(float)

        if df.empty:
            return pd.DataFrame(), None

        X = df[["TotalSpend", "NumOrders"]]
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        df["Cluster"] = model.fit_predict(X)
        return df, model

    # (9) Dự báo xu hướng doanh thu theo danh mục (ĐÃ SỬA LỖI NGÀY THÁNG)
    def predict_category_trend(self):
        sql = """
        SELECT YEAR(STR_TO_DATE(o.OrderDate, '%d/%m/%Y')) AS Year, c.Name AS CategoryName, -- Dùng định dạng DD/MM/YYYY
               SUM(od.OrderQty * od.UnitPrice) AS Revenue
        FROM orderdetails od
        JOIN orders o ON od.OrderID = o.OrderID
        JOIN product p ON od.ProductID = p.ProductID
        JOIN subcategory sc ON p.ProductSubcategoryID = sc.SubcategoryID 
        JOIN category c ON sc.CategoryID = c.CategoryID
        WHERE STR_TO_DATE(o.OrderDate, '%d/%m/%Y') IS NOT NULL
        GROUP BY YEAR(STR_TO_DATE(o.OrderDate, '%d/%m/%Y')), c.Name; -- GROUP BY đầy đủ
        """
        df = self.conn.queryDataset(sql)

        if df.empty:
            return {}

        # Ép kiểu sau khi query
        # Cần xử lý giá trị NaN/NULL từ DB sau khi GROUP BY
        df = df[df['Year'].notna()]
        if df.empty:
            return {}

        df["Year"] = df["Year"].astype(float)
        df["Revenue"] = df["Revenue"].fillna(0).astype(float)

        df = df.sort_values(["CategoryName", "Year"])
        results = {}

        for cat in df["CategoryName"].unique():
            sub = df[df["CategoryName"] == cat].copy()

            # Kiểm tra xem có đủ dữ liệu theo năm cho từng danh mục không
            if len(sub) < 2:
                results[cat] = {
                    "LatestYear": int(sub["Year"].max()) if not sub.empty else None,
                    "PredictedNextYear": None,
                    "ModelCoeff": None,
                    "ModelIntercept": None,
                    "Status": "Not enough data (need >= 2 years)"
                }
                continue

            X = sub[["Year"]]
            y = sub["Revenue"]
            model = LinearRegression()

            try:
                model.fit(X, y)

                next_year = X["Year"].max() + 1
                pred = model.predict(np.array([[next_year]]))  # Dùng np.array cho input

                results[cat] = {
                    "LatestYear": int(X["Year"].max()),
                    "PredictedNextYear": pred[0],
                    "ModelCoeff": model.coef_[0],
                    "ModelIntercept": model.intercept_,
                    "Status": "OK"
                }
            except Exception as e:
                results[cat] = {"Status": f"ML Error: {str(e)}"}

        return results