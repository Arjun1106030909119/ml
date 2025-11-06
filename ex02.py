import pandas as pd

# Read CSV file
df = pd.read_csv("student.csv")

# Display the DataFrame
df

df.describe()

df.info()

df.isnull().sum()

df.dtypes

df.shape

df1 = df.fillna("n")
df1

df2 = df.fillna(5)
df2

df1 = df.fillna({'chol': 1, 'fbs': 2})

# Check for missing values
df1.isnull().sum()
df1

# Carry forward (Forward fill)
df1 = df.fillna(method="ffill")
df1

# Backward fill
df1 = df.fillna(method="bfill")
df1

df1= df.interpolate()
df1

df1 = df.dropna()
df1
#csv
#reg_no , m1 , m2 , m3, result 
#100, 78 , 85 , 90, 0
#101, 88 , nan , 95,1
#102, nan , 89 , 84,0
#103, 75 , 80 , 88,1
#104, 85 , 90 , 92,1
#105, 80 , 78 , nan,0