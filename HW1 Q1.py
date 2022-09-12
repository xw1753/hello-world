#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

root = '/Users/wxq/MLSys-NYU-2022/weeks/2/data/'
data = pd.read_csv(root+'train.csv')
data.head()

X = data[['1stFlrSF','2ndFlrSF','TotalBsmtSF']].values
y = data[['SalePrice']].values

a = np.linalg.inv(np.transpose(X) @ X)
beta = (a @ np.transpose(X))@y
beta

y_pred = X.dot(beta)
y_pred

y_bar = y.mean()
y_bar

r_squared = 1 - ((y - y_pred)**2).sum() / ((y - y_bar)**2).sum() 
r_squared


