#csv file
#age,cholesterol,heart_disease
#young,high,0
#old,high,1
#young,normal,0
#old,high,1


import pandas as pd

df = pd.read_csv('data_ex06.csv')
print("Simple frequency-based probabilities:\n")

p_old = df[df['age']=='old']
print("P(heart_disease=1 | age=old) =", (p_old['heart_disease']==1).mean())
p_ch = df[df['cholesterol']=='high']
print("P(heart_disease=1 | cholesterol=high) =", (p_ch['heart_disease']==1).mean())
