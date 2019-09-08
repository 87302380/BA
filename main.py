from data import data
from parameters_optimization import hyperparameter_optimization

import format_time
import time

start = time.time()
path = "colon_label_in_first_row.csv"
data = data(path)

train_data = data.get_lgbDataset(377)

hpo = hyperparameter_optimization()
# default_loss = hpo.default_parameter(train_data, 62)

# bohb_para, bohb_loss = hpo.search_parameter_bohb(train_data, 62, 3, save=True)
# print(bohb_para)
# print(bohb_loss)

# rs_para, rs_loss = hpo.search_parameter_randomsearch(train_data, kfold=62,iterations= 3, save=True)
#
# print(rs_para)
# print(rs_loss)


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
for i in range(0,30):
    filepath = './result/bohb/loss_time_bohb' + str(i) + '.csv'
    bohb_para, bohb_loss = hpo.search_parameter_bohb(train_data, 62, 100, save=True, filepath=filepath)


print(time.time()-start)