"""
product
- id
- name
- quantity
- price
- image (nếu có)
"""
class Product:
    def __init__(self, id=None, name=None, quantity=None, price=None):
        self.id = id
        self.name = name
        self.quantity = quantity
        self.price = price
    """--str-- = tự động xuất đối tượng ra màn hình """
    def __str__(self):
        return f"{self.id}\t{self.name}\t{self.quantity}\t{self.price}"




