#!/bin/bash

# slice_learn_predict3.bash

# This script should slice iris.csv into /tmp/iris_train.csv and /tmp/iris_test.csv

# It should learn from /tmp/iris_train.csv
# It should predict iris_type values from values in /tmp/iris_test.csv
# It should compare predicted iris_type values from observed iris_type values in /tmp/iris_test.csv

python slice_train_test.py
python learn_predict_logr.py

exit


