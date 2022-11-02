# ML LIBRARIES
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.utils import column_or_1d
# DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
# ----------------------------------------------------------------------------------------------------------------------
import pickle
import numpy as np
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
import itertools
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# DATABASE LIBRARIES
import psycopg2
from psycopg2 import OperationalError
from psycopg2.extensions import register_adapter, AsIs


# ----------------------------------------------------------------------------------------------------------------------
# DATABASE CONNECTION FUNCTION
def create_connection(db_name, db_user, db_password, db_host, db_port):
    connection = None
    try:
        connection = psycopg2.connect(
            database=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
        )
        print("Connection to PostgreSQL DB successful")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    return connection


# ----------------------------------------------------------------------------------------------------------------------
# CONNECTING TO DATABASE
connection = create_connection("limitless_v1", "postgres", "Limitless@96", "127.0.0.1", "5432")
# ----------------------------------------------------------------------------------------------------------------------
# EXTRACTING EXISTING DATA
# vml_model_data = {}

select_data_database_query = f"""SELECT x_location,y_location,depth FROM head_data_chest_fly"""
# EXTRACTING TABLE NAMES
connection.autocommit = True
cursor = connection.cursor()
cursor.execute(select_data_database_query)
data_database_values = cursor.fetchall()
ml_model_data = data_database_values
ml_model_data = np.array(ml_model_data)

# print(ml_model_data)
# print(type(ml_model_data))
# ----------------------------------------------------------------------------------------------------------------------
X = ml_model_data[:, :2]
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

y = ml_model_data[:, 2:3].ravel()
y = y.astype('int')
y = column_or_1d(y, warn=True)

# X = preprocessing.StandardScaler().fit(X).transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
# ----------------------------------------------------------------------------------------------------------------------
# Train Model and Predict  - KNN
Ks = 10
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))

for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

print(mean_acc)

plt.plot(range(1, Ks), mean_acc, 'g')
plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1, Ks), mean_acc - 3 * std_acc, mean_acc + 3 * std_acc, alpha=0.10, color="green")
plt.legend(('Accuracy ', '+/- 1xstd', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print("The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax() + 1)
# ----------------------------------------------------------------------------------------------------------------------
clf = svm.SVC(gamma=0.001, C=100.)
regr = clf.fit(X_train, y_train)

# print('Coefficients: ', regr.coef_)

yhat = clf.predict(X_test)
# print(clf.predict(X_test))

# ERROR CHECK
print("Mean absolute error: %.2f" % np.mean(np.absolute(yhat - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((yhat - y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_test, yhat))
print('Variance score: %.2f' % regr.score(X_test, y_test))

# CONFUSION MATRIX
# print(confusion_matrix(y_test, yhat, labels=[0, 1]))
# s = pickle.dumps(clf)
