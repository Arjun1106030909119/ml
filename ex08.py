#f1,f2,target
#1,2,0
#1.1,2.0,0
#4,4.5,1
#4.2,4.6,1

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('data_ex08.csv')
X = df[['f1','f2']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
