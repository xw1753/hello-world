#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

root = '/Users/wxq/MLSys-NYU-2022/weeks/2/data/'
data = pd.read_csv(root+'train.csv')
data.head()

X = data[['1stFlrSF','2ndFlrSF','TotalBsmtSF','LotArea','OverallQual','GrLivArea','GarageCars','GarageArea']]
y = data[['SalePrice']].values

model = LinearRegression(fit_intercept=True)

metric = pd.DataFrame(columns = ['R_squared','MSE','MAE','MAPE'])
def df_metric(model, X, y):
    for i in range(1,9): 
        x = X.iloc[:,:i]
        model = model.fit(x,y)
        predictions = model.predict(x)
        R_squared = r2_score(y, predictions)
        MSE = mean_squared_error(y, predictions)
        MAE = mean_absolute_error(y, predictions)
        MAPE = mean_absolute_percentage_error(y, predictions)
        metric.loc[i] = [R_squared, MSE, MAE, MAPE]
        print('model', i ,'R_squared:', R_squared)
        print('MSE:', MSE)
        print('MAE:', MAE)
        print('MAPE:', MAPE)
    return metric

df_metric = df_metric(model, X, y)
df_metric

for i in range(0,4):
    fig, ax = plt.subplots()
    x = df_metric.index.values
    y = df_metric.iloc[:,i]
    ax.scatter(x, y, alpha=0.5)
    ax.set_xlabel("Number of Features")
    ax.set_ylabel(df_metric.columns[i])
    plt.show()




