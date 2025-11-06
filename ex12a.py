#feat1,feat2,label
#0.1,0.3,0
#0.9,0.8,1
#0.2,0.1,0
#0.8,0.7,1

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('data_ex12a.csv')
X = df[['feat1','feat2']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=0)
model = LogisticRegression()
model.fit(X_train, y_train)

print("Train acc:", accuracy_score(y_train, model.predict(X_train)))
print("Test acc:", accuracy_score(y_test, model.predict(X_test)))
