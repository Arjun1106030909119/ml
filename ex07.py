# ex07.py - EM / GaussianMixture example recreated from screenshot

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import numpy as np
import pandas as pd

# Loading data-set for EM algorithm
iris = datasets.load_iris()
X = pd.DataFrame(iris.data)
Y = pd.DataFrame(iris.target)

# Defining EM Model
from sklearn.mixture import GaussianMixture
model2 = GaussianMixture(n_components=3, random_state=3425)

# Training of the model
model2.fit(X)

# Predicting classes for our data
uu = model2.predict(X)

# Accuracy of EM Model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y, uu)
print(cm)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y, uu))