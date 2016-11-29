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

test_df = pd.read_csv('/tmp/iris_test.csv')
test_a = np.array(test_df[['f0','f2','f3','iris_type']])
predictions_l = linr_model.predict(test_a).tolist()

print(predictions_l)

'bye'

