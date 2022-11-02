import numpy as np
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
# DATABASE LIBRARIES
import psycopg2
from psycopg2 import OperationalError
# ----------------------------------------------------------------------------------------------------------------------
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# ----------------------------------------------------------------------------------------------------------------------
# ML Libraries
from sklearn.neighbors import KNeighborsClassifier
k = 4
# Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
select_data_database_query = f"""SELECT x_location,y_location,depth FROM head_data_chest_fly"""

connection.autocommit = True
cursor = connection.cursor()
cursor.execute(select_data_database_query)
data_database_values = cursor.fetchall()
ml_model_data = data_database_values
ml_model_data = np.array(ml_model_data)
# ----------------------------------------------------------------------------------------------------------------------
X = ml_model_data[:, :2]
y = ml_model_data[:, 2:3].ravel()
y = y.astype('int')

# plt.figure(figsize=(8,5))
# plt.plot(X, y, 'ro')
# plt.ylabel('GDP')
# plt.xlabel('Year')
# plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# CURVE FITTING MODEL
def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
    return y


# ----------------------------------------------------------------------------------------------------------------------
# SAMPLE TESTING
beta_1 = 0.10
beta_2 = 1990.0
# logistic function
y_pred = sigmoid(X, beta_1, beta_2)
# plot initial prediction against datapoints
# plt.plot(X, y_pred*15000000000000.)
# plt.plot(X, y, 'ro')
# plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# NORMALISED DATA
xdata = X[:, 1] / max(X[:, 1])
ydata = y / max(y)
# ----------------------------------------------------------------------------------------------------------------------
# CURVE FITTING
popt, pcov = curve_fit(sigmoid, xdata, ydata)
# FINAL PARAMETERS
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))
# ----------------------------------------------------------------------------------------------------------------------
# PLOT OF CURVE FIT
# x = np.linspace(0, 2500, 200)
# x = x / max(x)
x = xdata

# plt.figure(figsize=(8, 5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x, y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# EVALUATION

# print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
# print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
# from sklearn.metrics import r2_score
# print("R2-score: %.2f" % r2_score(test_y,y_hat))
