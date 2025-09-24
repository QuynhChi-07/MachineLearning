import pandas as pd

def get_orders_in_range(df, minValue, maxValue, SortType=True):
    """
    Trả về danh sách các hóa đơn có tổng giá trị trong khoảng [minValue, maxValue]
    và sắp xếp theo SortType.

    Parameters:
        df (pd.DataFrame): dữ liệu
        minValue (float): giá trị min
        maxValue (float): giá trị max
        SortType (bool): True = tăng dần, False = giảm dần

    Returns:
        pd.DataFrame: gồm OrderID và tổng giá trị
    """

    # Tính tổng giá trị theo OrderID
    order_totals = df[['OrderID', 'UnitPrice', 'Quantity', 'Discount']].groupby('OrderID').apply(
        lambda x: (x['UnitPrice'] * x['Quantity'] * (1 - x['Discount'])).sum()
    ).reset_index(name='Sum')

    # Lọc theo khoảng [minValue, maxValue]
    filtered = order_totals[(order_totals['Sum'] >= minValue) & (order_totals['Sum'] <= maxValue)]

    # Sắp xếp theo SortType
    filtered = filtered.sort_values(by='Sum', ascending=SortType).reset_index(drop=True)

    return filtered


# --- Ví dụ chạy ---
df = pd.read_excel('../dataset/SalesTransactions/SalesTransactions.xlsx')

minValue = float(input("Nhập giá trị min: "))
maxValue = float(input("Nhập giá trị max: "))
SortType = True  # Đổi thành False để sắp xếp giảm dần

result = get_orders_in_range(df, minValue, maxValue, SortType)
print(result)
