from data import data
from parameters_optimization import hyperparameter_optimization
from sklearn.model_selection import LeaveOneOut
import lightgbm as lgb

import time

start = time.time()
path = "colon_label_in_first_row.csv"
data = data(path)

input_data, test_data = data.spilt_data()

train_data = data.get_lgbDataset(377)

hpo = hyperparameter_optimization()
feature_loo = LeaveOneOut()
for excluded_feature_index, target_feature_index in feature_loo.split(range(input_data.shape[1])):

    selected_x = input_data[:, excluded_feature_index]
    selected_y = input_data[:, target_feature_index][:, 0]
    train_data = lgb.Dataset(selected_x, label= selected_y)

    param, loss = hpo.default_parameter(target_feature_index, train_data, 46)
    param, loss = hpo.search_parameter_hyperband(target_feature_index, train_data, 46, 10)


# for i in range(0, 30):
#     filepath = './result/gs/loss_time_gs' + str(i) + '.csv'
#     loss = hpo.search_parameter_gs(377, train_data, 10, 2, save=True, filepath=filepath)

