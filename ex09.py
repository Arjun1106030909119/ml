#a,b,c,target
#1,0,1,0
#1,1,0,1
#0,1,1,1
#0,0,1,0
#1,1,1,1

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('data_ex09.csv')
X = df[['a','b','c']]
y = df['target']

clf_unpruned = DecisionTreeClassifier(random_state=0)
clf_pruned = DecisionTreeClassifier(max_depth=2, random_state=0)

clf_unpruned.fit(X, y)
clf_pruned.fit(X, y)

print("Unpruned depth:", clf_unpruned.get_depth())
print("Pruned depth:", clf_pruned.get_depth())
print("Unpruned accuracy:", accuracy_score(y, clf_unpruned.predict(X)))
print("Pruned accuracy:", accuracy_score(y, clf_pruned.predict(X)))
