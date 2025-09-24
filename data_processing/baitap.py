"""
Viết 1 hàm có mô tả:
Input:
-DataFrame => df
-Tổng Giá Trị Min =>minValue
-Tổng Giá Trị Max =>maxValue
Output:
-Trả về danh sách các hóa đơn (mã hóa đơn) mà tổng trị giá của nó nằm trong [minValue …maxValue]
"""
import pandas as pd

def find_orders_within_range(df,minValue,maxValue):
    # tổng giá trị từng đơn hàng
    order_totals = df.groupby('OrderID').apply(lambda x: (x['UnitPrice'] * x['Quantity'] * (1 - x['Discount'])).sum())
    #lọc đơn hàng trong range
    orders_with_range=order_totals[(order_totals>=minValue) & (order_totals<=maxValue)]
    #danh sách caác mã đơn hàng không trùng nhau
    unique_orders = df[df['OrderID'].isin(orders_with_range.index)]['OrderID'].drop_duplicates().tolist()

    return unique_orders

df=pd.read_csv('../dataset/SalesTransactions/SalesTransactions.csv')

minValue=float(input("Nhập giá trị min:"))
maxValue=float(input("Nhập giá trị max:"))
result=find_orders_within_range(df,minValue,maxValue)
print("Danh sách các hoá đơn trong phạm vi giá trị từ", minValue, "đến", maxValue, "là", result)




