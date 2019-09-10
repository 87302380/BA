import LightGBMWorker_management_BOHB as bohb
import LightGBMWorker_management_RS as rs
import tpe
import anneal
import Optuna
import gridsearch
import object_function
import pandas as pd

class hyperparameter_optimization:

    def __init__(self):
        self.dict = {}

    def default_parameter(self, target_feature_index, train_data, kfold):
        para = {}
        parameter = self.format_parameter(para)
        loss = object_function.cv_method(parameter, train_data, kfold)
        self.update_dict(target_feature_index, parameter, loss)
        return parameter, loss

    def search_parameter_gridsearch(self, target_feature_index, train_data, kfold, param_grid = None):
        if param_grid:
            parameter, loss = gridsearch.run_grid_search(train_data, kfold, param_grid)
            self.update_dict(target_feature_index, parameter, loss)
        else:
            parameter, loss = gridsearch.run_grid_search(train_data, kfold)
            self.update_dict(target_feature_index, parameter, loss)
        return parameter, loss

    def search_parameter_randomsearch(self, target_feature_index, train_data, kfold, iterations, save=False, filepath = './result/loss_time_rs.csv'):
        params, loss = rs.get_parameters(train_data, kfold, iterations, save=save, filepath = filepath)
        parameter = self.format_parameter(params)
        self.update_dict(target_feature_index, params, loss)

        return parameter, loss

    def search_parameter_tpe(self, target_feature_index, train_data, kfold, iterations, save=False, filepath = './result/loss_time_tpe.csv'):
        parameter, loss = tpe.get_parameters(train_data, kfold, iterations, save=save, filepath=filepath)
        self.update_dict(target_feature_index, parameter, loss)

        return parameter, loss

    def search_parameter_anneal(self, target_feature_index, train_data, kfold, iterations, save=False, filepath = './result/loss_time_anneal.csv'):
        parameter, loss = anneal.get_parameters(train_data, kfold, iterations, save=save, filepath=filepath)
        self.update_dict(target_feature_index, parameter, loss)

        return parameter, loss

    def search_parameter_bohb(self, target_feature_index, train_data, kfold, iterations, save=False, filepath = './result/loss_time_bohb.csv'):
        parameter, loss = bohb.get_parameters(train_data, kfold, iterations, save=save, filepath = filepath)
        parameter = self.format_parameter(parameter)
        self.update_dict(target_feature_index, parameter, loss)

        return parameter, loss

    def search_parameter_optuna(self, target_feature_index, train_data, kfold, iterations, save=False, filepath = './result/loss_time_optuna.csv'):
        parameter, loss = Optuna.get_parameters(train_data, kfold, iterations, save=save, filepath = filepath)
        parameter = self.format_parameter(parameter)
        self.update_dict(target_feature_index, parameter, loss)

        return parameter, loss

    def update_dict(self, target_feature_index, params, loss):
        info = {}
        info['parameter'] = params
        info['loss'] = loss
        self.dict[int(target_feature_index)] = info
        result = pd.DataFrame.from_dict(self.dict, orient='index')
        result.to_csv('./result/result.csv')

    def get_dict(self):
        return self.dict

    def get_feature_info(self, key):
        return self.dict.get(key)



    def format_parameter(self, params):
        params['boosting_type'] = 'gbdt'
        params['objective'] = 'regression_l2'
        params['verbose'] = -1

        return params








