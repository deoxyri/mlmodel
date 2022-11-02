import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from pygments.lexers import graphviz
from scipy.optimize import curve_fit

from sklearn.model_selection import train_test_split
from sklearn import svm

from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
# METRICS LIBRARIES
from sklearn.metrics import classification_report, confusion_matrix, log_loss, f1_score
from sklearn.metrics import jaccard_score
# ----------------------------------------------------------------------------------------------------------------------
# DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING
import pydotplus

# ----------------------------------------------------------------------------------------------------------------------
# DATA
cell_df = pd.read_csv("cell_samples.csv")
print(cell_df.head())
print(cell_df.shape)
print(cell_df.dtypes)

ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue',
                                               label='malignant')
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign',
                                          ax=ax)
# plt.show()

cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
print(cell_df.dtypes)
# ----------------------------------------------------------------------------------------------------------------------
# DATA EXTRACTION
feature_df = cell_df[
    ['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)

cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
# ----------------------------------------------------------------------------------------------------------------------
# DATA SPLITTING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)
# ----------------------------------------------------------------------------------------------------------------------
# SVM MODEL - RADIAL BASIS FUNCTION
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)


# ----------------------------------------------------------------------------------------------------------------------
# CONFUSION MATRIX PLOT FUNCTION
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


# ----------------------------------------------------------------------------------------------------------------------
# COMPUTE CONFUSION MATRIX
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2, 4])
np.set_printoptions(precision=2)
print(classification_report(y_test, yhat))

# PLOT CM
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)', 'Malignant(4)'], normalize=False, title='Confusion matrix')
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# METRICS
print(f1_score(y_test, yhat, average='weighted'))
print(jaccard_score(y_test, yhat, pos_label=2))
# ----------------------------------------------------------------------------------------------------------------------
# SVM - LINEAR MODEL
clf2 = svm.SVC(kernel='linear')
clf2.fit(X_train, y_train)
yhat2 = clf2.predict(X_test)
print("Avg F1-score: %.4f" % f1_score(y_test, yhat2, average='weighted'))
print("Jaccard score: %.4f" % jaccard_score(y_test, yhat2,pos_label=2))
