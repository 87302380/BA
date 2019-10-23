from data import data
from parameters_optimization import hyperparameter_optimization

# Set the path to the data file
path = "colon_label_in_first_row.csv"

# Load data using the data class
data = data(path)

# The data class provides the ability to split data, which can be easily divided into test sets and training sets.
input_data, test_data = data.spilt_data()

# can get the dataset that matches leaveoneout directly by using the get_lgbDataset() method.
train_data = data.get_lgbDataset(377)

# Create a hyperparameter optimized object, use this object to easily use different hyperparameter optimization methods.
hpo = hyperparameter_optimization()

# Repeat the experiment. We repeated 30 times for all methods, and some methods were repeated 60 times.
for i in range(0, 30):

    # Set the output path of the result
    filepath = './result/gs/loss_time_optuna' + str(i) + '.csv'

    # Choosing a method of hyperparameter optimization
    loss = hpo.search_parameter_optuna(target_feature_index=377, train_data=train_data, kfold=10, iterations=2, save=True, filepath=filepath)



