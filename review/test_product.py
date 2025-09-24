from review.product import Product

p1=Product(100, "Thuốc xinh đẹp", 4, 20)
print(p1)
#chỉ ra địa chỉ ô nhớ do chưa định nghĩa str

p2=Product(200, "Thuốc giảm cân", 5, 30)
"""
p1 ở ô nhớ alpha
p2 ở ô nhớ beta
p1=p2 --> p1 trỏ tới ô nhớ p2 ==> p1 ở cùng p2ở ô beta
"""
p1=p2
print("Thông tin của p1 =")
print (p1)
p1.name = "Thuốc tăng mỡ"
print ("Thông tin của p2 =")
print (p2)