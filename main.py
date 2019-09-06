from data import data
from parameters_optimization import hyperparameter_optimization

import format_time
import time

path = "colon_label_in_first_row.csv"
data = data(path)

train_data = data.get_lgbDataset(377)

hpo = hyperparameter_optimization()
start = time.time()
# default_loss = hpo.default_parameter(train_data, 62)

# bohb_para, bohb_loss = hpo.search_parameter_bohb(train_data, 62, 10)
# print(bohb_para)
# print(bohb_loss)

# rs_para, rs_loss = hpo.search_parameter_randomsearch(train_data, kfold=62,iterations= 10)
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

tpe, tpe_loss = hpo.search_parameter_anneal(train_data, kfold=62, iterations=1, save=True)

print(tpe)
print(tpe_loss)

# optuna_para, optuna_loss = hpo.search_parameter_optuna(train_data, kfold=62, iterations=10)
#
# print(optuna_para)
# print(optuna_loss)


format_time.print_time(start)