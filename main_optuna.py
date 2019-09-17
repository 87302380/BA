from data import data
from parameters_optimization import hyperparameter_optimization

import format_time
import time

start = time.time()
path = "/home/lchen/BA/colon_label_in_first_row.csv"
data = data(path)

train_data = data.get_lgbDataset(377)

hpo = hyperparameter_optimization()
# default_loss = hpo.default_parameter(train_data, 62)


# rs_para, rs_loss = hpo.search_parameter_randomsearch(train_data, kfold=62,iterations= 3, save=True)
#
# print(rs_para)
# print(rs_loss)


# tpe, tpe_loss = hpo.search_parameter_tpe(train_data, kfold=62, iterations=3, save=True)
#
# print(tpe)
# print(tpe_loss)
for i in range(0,30):
    filepath = '/vol/projects/lchen/result/optuna/loss_time_optuna' + str(i) + '.csv'
    hpo.search_parameter_optuna(377, train_data, kfold=62, iterations=200, save=True, filepath=filepath)

