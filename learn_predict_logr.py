# learn_predict_logr.py

import pandas as pd
import numpy  as np

# This script should learn from /tmp/iris_train.csv
# I should assume that iris_type column depends on columns: f0,f1,f2,f3

train_df = pd.read_csv('/tmp/iris_train.csv')

# I should collect independent variables in a nested list:
x_a = np.array(train_df[['f0','f1','f2','f3']])

# I should import linear_model:
from sklearn import linear_model
logr_model = linear_model.LogisticRegression()
# I should call fit() to create the model:
logr_model.fit(x_a, train_df.iris_type)

# I now have a model inside logr_model.
# I should test the model.
test_df       = pd.read_csv('/tmp/iris_test.csv')
# sklearn models prefer numpy arrays:
test_a        = np.array(test_df[['f0','f1','f2','f3']])
# sklearn prefers to give predictions as numpy arrays:
predictions_a = np.round(logr_model.predict_proba(test_a),2)

# I should capture the predictions for reporting.
# I should report by adding columns to test_df and then printing it:
predictions_l             = [row for row in predictions_a]
test_df['predictions']    = predictions_l
predicted_types_l         = [np.ndarray.argmax(row) for row in predictions_a]
test_df['predicted_type'] = predicted_types_l
test_df['accurate']       = (test_df.iris_type == test_df.predicted_type)
accuracy_f                = 100.0*np.sum(test_df.accurate)/len(test_df)

# I should report:
print(test_df)
print('percent accuracy:')
print(accuracy_f)

'bye'

