class ListProduct:
    def __init__(self):
        self.products = []
    def add_Product(self, p):
        self.products.append(p)
    def print_products(self):
        for p in self.products:
            print(p)
    def desc_sort_products(self):
        for i in range (0, len(self.products)):
            for j in range (i+1, len(self.products)):
                pi=self.products[i]
                pj=self.products[j]
                if pi.price < pj.price:
                    self.products[j]=pi
                    self.products[i]=pj
    # Sắp xếp giảm dần theo price (không dùng vòng lặp thủ công)
    def desc_sort_products_no_for(self):
        result = []
        while self.products:
        # Tìm phần tử lớn nhất
           max_p = max(self.products, key=lambda p: p.price)
           result.append(max_p)
           self.products.remove(max_p)
        self.products = result

    def _recursive_sort(self, products):
        if not products:
            return []
        # Tìm sản phẩm có giá lớn nhất
        max_p = max(products, key=lambda p: p.price)
        products.remove(max_p)
        # Đệ quy cho phần còn lại
        return [max_p] + self._recursive_sort(products)


