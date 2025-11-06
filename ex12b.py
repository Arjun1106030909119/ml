#preg,glucose,bp,diabetes
#1,85,66,0
#3,145,82,1
#2,90,70,0
#4,160,85,1

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('data_ex12b.csv')
X = df[['preg','glucose','bp']]
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=1)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

print("Test accuracy:", accuracy_score(y_test, model.predict(X_test)))
print("\nClassification report:\n", classification_report(y_test, model.predict(X_test)))
