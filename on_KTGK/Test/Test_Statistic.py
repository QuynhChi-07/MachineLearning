import pandas as pd
# Giả định Connector nằm trong on_KTGK.Connectors
from on_KTGK.Connectors.Connector import Connector
from on_KTGK.Models.StatisticModel import StatisticModel

# --- KHỞI TẠO VÀ CHẠY ---

# 1. Khởi tạo và kết nối Connector
db_connector = Connector()
conn = db_connector.connect()

if conn:
    # 2. Khởi tạo StatisticModel, truyền đối tượng connector vào
    stat = StatisticModel(db_connector)

    print("Tổng Doanh Số Bán Hàng ---")
    print(stat.total_sales())

    print("Doanh Thu Theo Danh Mục ---")
    print(stat.revenue_by_category())

    # 3. Sửa tên hàm: early_deliveries() -> fast_delivery_orders()
    print("Đơn Hàng Giao Nhanh ---")
    print(stat.fast_delivery_orders())

    # Ngắt kết nối sau khi hoàn tất
    db_connector.disConnect()
else:
    print("[LỖI] Không thể chạy test vì kết nối database thất bại.")
