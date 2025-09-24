import sqlite3
import pandas as pd

def get_top_customers_by_invoices(db_path, top_n):
    try:
        # Kết nối DB
        sqliteConnection = sqlite3.connect(db_path)
        cursor = sqliteConnection.cursor()
        print("DB Init")

        # Query: đếm số Invoice + tính tổng trị giá
        query = f"""
        SELECT 
            c.CustomerId AS CustomerId,
            c.FirstName || ' ' || c.LastName AS FullName,
            COUNT(i.InvoiceId) AS InvoiceCount,
            SUM(i.Total) AS TotalValue
        FROM Customer c
        JOIN Invoice i ON c.CustomerId = i.CustomerId
        GROUP BY c.CustomerId
        ORDER BY InvoiceCount DESC, TotalValue DESC
        LIMIT {top_n};
        """

        cursor.execute(query)

        # Lấy dữ liệu
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]

        # Đưa vào DataFrame
        df = pd.DataFrame(rows, columns=col_names)
        return df

    except sqlite3.Error as error:
        print("Error occurred:", error)
        return None
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("sqlite connection closed")

# --- Main ---
if __name__ == "__main__":
    db_path = "../databases/databases/Chinook_Sqlite.sqlite"
    top_n = int(input("Nhập số khách hàng cần lấy (TOP N): "))
    result = get_top_customers_by_invoices(db_path, top_n)
    print(f"Top {top_n} khách hàng có số Invoice nhiều nhất:")
    print(result)
