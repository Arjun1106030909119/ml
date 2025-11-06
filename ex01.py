import pandas as pd

# Read CSV file
df = pd.read_csv("student.csv")

# Display the DataFrame
df

# Display first two rows
df.head(2)

# Display last two rows
df.tail(2)


#csv
#reg_no , m1 , m2 , m3, result 
#100, 78 , 85 , 90, 0
#101, 88 , nan , 95,1
#102, nan , 89 , 84,0
#103, 75 , 80 , 88,1
#104, 85 , 90 , 92,1
#105, 80 , 78 , nan,0