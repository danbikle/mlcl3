# learn_predict_linr.py

import pandas as pd
import numpy  as np
import pdb
# This script should learn from /tmp/iris_train.csv
# I should assume that f1 column depends on columns: f0,f2,f3,iris_type

train_df = pd.read_csv('/tmp/iris_train.csv')

# I should collect independent variables in a nested list:
x_a = np.array(train_df[['f0','f2','f3','iris_type']])

# I should import linear_model:
from sklearn import linear_model
linr_model = linear_model.LinearRegression()
# I should call fit() to create the model:
linr_model.fit(x_a, train_df.f1)

# I should save the model:

w0_f = linr_model.intercept_
w_l  = linr_model.coef_
m_l  = [w0_f] + w_l.tolist()
m_sr = pd.Series(m_l)
m_s  = '/tmp/iris_model.csv'
m_sr.to_csv(m_s, float_format='%4.2f', index=False, header=['weights'])
print('Model saved to: '+ '/tmp/iris_model.csv')
'bye'

