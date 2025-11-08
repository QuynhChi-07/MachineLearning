class Customer:
    def __init__(self, customer_id, name, title, person_type): # Điều chỉnh các tham số
        self.customer_id = customer_id
        self.name = name
        self.title = title
        self.person_type = person_type

    @staticmethod
    def from_row(row):
        # Tạo tên đầy đủ
        full_name = f"{row['FirstName']} {row['LastName']}"
        # Giả định row là kết quả từ fetchone/fetchall (dạng dict)
        return Customer(
            row["CustomerID"],
            full_name.strip(),
            row["Title"],
            row["PersonType"]
        )