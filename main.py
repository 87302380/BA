from data import data
from parameters_optimization import hyperparameter_optimization
from sklearn.model_selection import LeaveOneOut
import lightgbm as lgb

import time

start = time.time()
path = "colon_label_in_first_row.csv"
data = data(path)

input_data, test_data = data.spilt_data()


hpo = hyperparameter_optimization()
feature_loo = LeaveOneOut()
for excluded_feature_index, target_feature_index in feature_loo.split(range(input_data.shape[1])):

    selected_x = input_data[:, excluded_feature_index]
    selected_y = input_data[:, target_feature_index][:, 0]
    train_data = lgb.Dataset(selected_x, label= selected_y)

    # param, loss = hpo.default_parameter(target_feature_index, train_data, 46)
    param, loss = hpo.search_parameter_optuna(target_feature_index, train_data, 46, 10)


#
# y_predicted = booster.predict(x_test)
#
# print(np.round(y_predicted))
# print(y_test)
# squared_error = (y_predicted - y_test) * (y_predicted - y_test)
# print(squared_error)
# pg = {
#     'boosting_type': ['gbdt'],
#     'objective': ['regression_l2'],
#     'num_leaves': [10, 31, 64, 127],
# }
#
# gs, gs_loss = hpo.search_parameter_gridsearch(train_data, kfold=10, param_grid= pg)
# print(gs)
# print(gs_loss)

# tpe, tpe_loss = hpo.search_parameter_tpe(train_data, kfold=62, iterations=3, save=True)
#
# print(tpe)
# print(tpe_loss)
# loss = hpo.default_parameter(train_data, 62)
# print(loss)
#
# for i in range(0,1):
#     filepath = './result/anneal/loss_time_anneal_test' + str(i) + '.csv'
#     hpo.search_parameter_anneal(train_data, 62, 40, save=True, filepath=filepath)
#
#
# print(time.time()-start)