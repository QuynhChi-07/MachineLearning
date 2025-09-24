#Filtering: loại bỏ data theo đề bài

from numpy import nan as NA
import pandas as pd
data = pd.DataFrame([[1., 6.5, 3.],
                     [1., NA, NA],
                     [NA, NA, NA],
                     [NA, 6.5, 3.]])

print(data)
print("-"*10)
cleaned = data.dropna()
#dropna: loại bỏ những dòng có ít nhất 1 gtri NULL = chỉ giữ lại những dòng ko có NaN nào cả

print (cleaned)
cleaned2=data.dropna(how='all') #xoá tất cả dòng full NA
print (cleaned2)

#Tìm hiểu về xoá dòng có dấu âm

