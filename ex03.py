import pandas as pd

data = pd.DataFrame({
    "A":[1,2,3,4,5,6],
    "B":[7,8,9,10,11,12],
    "C":[0,0,0,0,0,0],
    "D":[21,54,32,85,35,2]
})

data

from sklearn.feature_selection import VarianceThreshold

var_thres = VarianceThreshold(threshold = 0)

var_thres.fit(data)

var_thres.get_support()

data.columns[var_thres.get_support()]

constant_columns = [column for column in data.columns if column not in data.columns[var_thres.get_support()]]

print(len(constant_columns))

for feature in constant_columns: print(feature)

data.drop(constant_columns, axis = 1)