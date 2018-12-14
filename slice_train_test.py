"""
slice_train_test.py

This script should slice iris.csv into /tmp/iris_train.csv and /tmp/iris_test.csv

Demo:
python slice_train_test.py
"""
import pandas as pd

# I should read iris.csv into a DataFrame:

iris0_df = pd.read_csv('iris.csv')

# I should randomize rows
import sklearn
iriss_df = sklearn.utils.shuffle(iris0_df).reset_index()
# Now I should have an extra column named index (which I dont need)

# I should encode iris_type column as integers: 0,1,2

iris_type_i_l = []
for type_i in iriss_df.iris_type:
    if   type_i == 'setosa':
        iris_type_i_l.append(0)
    elif type_i == 'versicolor':
        iris_type_i_l.append(1)
    else:
        iris_type_i_l.append(2)

iris1_df = iriss_df.copy()[['f0','f1','f2','f3']]
iris1_df['iris_type'] = iris_type_i_l

# I should get the training data:

train_df = iris1_df.iloc[0:140]

# I should get the test data:

test_df = iris1_df.iloc[140:150]

# I should write to csv files:

train_df.to_csv('/tmp/iris_train.csv', float_format='%4.2f', index=False)
test_df.to_csv( '/tmp/iris_test.csv' , float_format='%4.2f', index=False)

print('Train Data should be here: /tmp/iris_train.csv')
print('Test  Data should be here: /tmp/iris_test.csv' )

'bye'
