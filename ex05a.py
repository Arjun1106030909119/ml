#csv file
#Pclass,Sex,Age,Survived
#3,male,22,0
#1,female,38,1
#3,female,26,1
#1,male,35,1
#2,male,28,0


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv('data_ex05a.csv')
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
X = df[['Pclass','Sex','Age']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Train accuracy:", accuracy_score(y_train, model.predict(X_train)))
print("Test accuracy:", accuracy_score(y_test, y_pred))
cv_scores = cross_val_score(model, X, y, cv=3)
print("CV scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())
