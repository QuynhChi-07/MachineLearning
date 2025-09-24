from review.product import Product
from review.products import ListProduct

lp=ListProduct()
lp.add_Product(Product(100, "Product 1", 200, 10))
lp.add_Product(Product(200, "Product 2", 10, 15))
lp.add_Product(Product(150, "Product 3", 80, 8))
lp.add_Product(Product(300, "Product 4", 50, 20))
lp.add_Product(Product(250, "Product 5", 150, 17))
print("List of Products:")
lp.print_products()

lp.desc_sort_products()
"sắp xếp sp theo đơn giá giảm dần - dùng 2 vòng lặp lòng nhau"
print("List of Products after descending sort:")
lp.print_products()

"sắp xếp ko dùng vòng lặp"
lp.desc_sort_products_no_for()
print ("List of Products after descending sort:")
lp.print_products()

