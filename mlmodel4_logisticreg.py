import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from pygments.lexers import graphviz
from scipy.optimize import curve_fit

from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
# METRICS LIBRARIES
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.metrics import jaccard_score
# ----------------------------------------------------------------------------------------------------------------------
# DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING
import pydotplus
# ----------------------------------------------------------------------------------------------------------------------
# READING DATA
churn_df = pd.read_csv("ChurnData.csv", delimiter=",")
print(churn_df.head())
# ----------------------------------------------------------------------------------------------------------------------
# PREPROCESSING
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
print(churn_df.head())

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X = preprocessing.StandardScaler().fit(X).transform(X)

y = np.asarray(churn_df['churn'])
# ----------------------------------------------------------------------------------------------------------------------
# TRAIN/TEST DATASET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)
# ----------------------------------------------------------------------------------------------------------------------
# MODEL
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
yhat = LR.predict(X_test)
# PROBABILITY OF CLASS
yhat_prob = LR.predict_proba(X_test)
print(yhat_prob)

print(jaccard_score(y_test, yhat, pos_label=0))


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# CONFUSION MATRIX
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# print(confusion_matrix(y_test, yhat, labels=[1,0]))

# COMPUTE
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1, 0])
np.set_printoptions(precision=2)
# PLOT NON-NORMALIZED CM
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1', 'churn=0'], normalize=False, title='Confusion matrix')
# plt.show()

print(classification_report(y_test, yhat))
print("LogLoss: : %.2f" % log_loss(y_test, yhat_prob))


# LOGREG 2
LR2 = LogisticRegression(C=0.01, solver='sag').fit(X_train, y_train)
yhat_prob2 = LR2.predict_proba(X_test)
print("LogLoss: : %.2f" % log_loss(y_test, yhat_prob2))
