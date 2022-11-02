import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygments.lexers import graphviz
from scipy.optimize import curve_fit
from sklearn import preprocessing
from sklearn import metrics
# ----------------------------------------------------------------------------------------------------------------------
# DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn.model_selection import train_test_split
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING
import pydotplus

# import
# ----------------------------------------------------------------------------------------------------------------------
my_data = pd.read_csv("drug200.csv", delimiter=",")

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

print(X)
# ----------------------------------------------------------------------------------------------------------------------
# DATA CONVERSION: CATEGORICAL TO DUMMY
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:, 1] = le_sex.transform(X[:, 1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 2] = le_BP.transform(X[:, 2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = le_Chol.transform(X[:, 3])
# ----------------------------------------------------------------------------------------------------------------------
y = my_data["Drug"]
# ----------------------------------------------------------------------------------------------------------------------
# SPLITTING DATA
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
# ----------------------------------------------------------------------------------------------------------------------
# MODELLING THE DECISION TREE
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree.fit(X_trainset, y_trainset)

predTree = drugTree.predict(X_testset)
print(predTree[0:5])
print(y_testset[0:5])
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
# ----------------------------------------------------------------------------------------------------------------------
# VISUALISATION
tree.plot_tree(drugTree)
plt.show()
