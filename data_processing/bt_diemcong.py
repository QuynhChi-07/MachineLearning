"""
Input: df
Output: Top 3 sản phẩm bán ra có giá trị lớn nhất (sản phẩm bán chạy nhất)
"""
import pandas as pd
def top3_best_selling_products(df):
    """
    Trả về Top 3 sản phẩm bán ra có giá trị lớn nhất.
    Input:
        df (pd.DataFrame): dữ liệu bán hàng
    Output:
        pd.DataFrame: gồm ProductID và tổng giá trị bán ra
    """
    # Tính tổng doanh thu từng sản phẩm
    product_sales = df[['ProductID', 'UnitPrice', 'Quantity', 'Discount']].groupby('ProductID').apply(
        lambda x: (x['UnitPrice'] * x['Quantity'] * (1 - x['Discount'])).sum()
    ).reset_index(name='TotalSales')

    # Lấy top 3 sản phẩm có doanh thu lớn nhất
    top3 = product_sales.sort_values(by='TotalSales', ascending=False).head(3).reset_index(drop=True)
    return top3

df = pd.read_excel('../dataset/SalesTransactions/SalesTransactions.xlsx')

result = top3_best_selling_products(df)
print("Top 3 sản phẩm bán chạy nhất:")
print(result)
