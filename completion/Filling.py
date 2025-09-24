from numpy import nan as NA
import pandas as pd
data = pd.DataFrame([[1., 6.5, 3.],
                     [2., NA, NA],
                     [NA, NA, NA],
                     [3, 6.5, 3.],
                     [4, 7.5, 7],
                     [5, 2.5, 3.],
                     [NA, NA, NA],
                     [6, 6.5, 3.]
                     ])

print(data)
print("-"*10)
cleaned = data.fillna(data.mean())
#fillna: fill tất cả gtri NaN

print (cleaned)