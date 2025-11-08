import pandas as pd
# Giả định các hàm trong connector trả về DataFrame (queryDataset)
# hoặc tuple/dict (fetchone)

class StatisticModel:
    def __init__(self, connector):
        # Lưu đối tượng connector để thực hiện truy vấn
        self.conn = connector

    # (2) Tổng doanh số bán hàng (số lượng * giá)
    def total_sales(self):
        sql = """
        SELECT 
            p.Name AS ProductName,                         -- SỬA: ProductName -> Name
            SUM(od.OrderQty * od.UnitPrice) AS TotalSales  -- SỬA: Quantity -> OrderQty
        FROM orderdetails od                               -- SỬA: orderdetails
        JOIN product p ON od.ProductID = p.ProductID       -- SỬA: products -> product
        GROUP BY p.Name
        ORDER BY TotalSales DESC;
        """
        # Giả định queryDataset là hàm thực hiện và trả về kết quả
        return self.conn.queryDataset(sql)

    # (3) Tổng doanh thu theo danh mục
    def revenue_by_category(self):
        sql = """
        SELECT 
            c.Name AS CategoryName,                        -- SỬA: CategoryName -> Name
            SUM(od.OrderQty * od.UnitPrice) AS TotalRevenue -- SỬA: Quantity -> OrderQty
        FROM orderdetails od
        JOIN product p ON od.ProductID = p.ProductID
        JOIN subcategory sc ON p.ProductSubcategoryID = sc.SubcategoryID -- Bổ sung JOIN trung gian
        JOIN category c ON sc.CategoryID = c.CategoryID                  -- SỬA: categories -> category
        GROUP BY c.Name
        ORDER BY TotalRevenue DESC;
        """
        return self.conn.queryDataset(sql)

    # (4) Doanh thu theo danh mục, Tháng + Năm
    def revenue_by_category_month_year(self):
        sql = """
        SELECT 
            c.Name AS CategoryName,                        -- SỬA: CategoryName -> Name
            MONTH(o.OrderDate) AS Month,
            YEAR(o.OrderDate) AS Year,
            SUM(od.OrderQty * od.UnitPrice) AS Revenue     -- SỬA: Quantity -> OrderQty
        FROM orderdetails od
        JOIN orders o ON od.OrderID = o.OrderID
        JOIN product p ON od.ProductID = p.ProductID
        JOIN subcategory sc ON p.ProductSubcategoryID = sc.SubcategoryID -- Bổ sung JOIN trung gian
        JOIN category c ON sc.CategoryID = c.CategoryID                  -- SỬA: categories -> category
        GROUP BY c.Name, MONTH(o.OrderDate), YEAR(o.OrderDate)
        ORDER BY Year, Month;
        """
        return self.conn.queryDataset(sql)

    # (5) Các đơn hàng giao nhanh hơn 3 ngày
    def fast_delivery_orders(self):
        sql = """
        SELECT 
            OrderID, CustomerID, OrderDate, ShipDate,      -- SỬA: ShippedDate -> ShipDate
            DATEDIFF(DueDate, ShipDate) AS DaysEarly       -- DATEDIFF: dùng DueDate và ShipDate
        FROM orders
        WHERE DATEDIFF(DueDate, ShipDate) >= 3;            -- So sánh với DueDate (Ngày phải giao)
        """
        return self.conn.queryDataset(sql)

    # (6) Chi tiết Customer
    def customer_detail(self, customer_id):
        sql = "SELECT * FROM customer WHERE CustomerID = %s;" # SỬA: customers -> customer
        # Giả định fetchone() trả về 1 kết quả
        return self.conn.fetchone(sql, (customer_id,))

    # (7) Tất cả đơn hàng của Customer
    def orders_by_customer(self, customer_id):
        sql = """
        SELECT o.OrderID, o.OrderDate, o.ShipDate, o.Freight,  -- SỬA: ShippedDate -> ShipDate
               SUM(od.OrderQty * od.UnitPrice) AS TotalOrderValue -- SỬA: Quantity -> OrderQty
        FROM orders o
        JOIN orderdetails od ON o.OrderID = od.OrderID
        WHERE o.CustomerID = %s
        GROUP BY o.OrderID, o.OrderDate, o.ShipDate, o.Freight;
        """
        # Giả định queryDataset() chấp nhận tham số
        return self.conn.queryDataset(sql, (customer_id,))