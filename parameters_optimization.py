import LightGBMWorker_management_BOHB as bohb
import LightGBMWorker_management_RS as rs
import tpe
import anneal
import Optuna
import gridsearch
import object_function

class hyperparameter_optimization:

    def default_parameter(self, train_data, kfold):
        para = {}
        parameter = self.format_parameter(para)
        loss = object_function.cv_method(parameter, train_data, kfold)

        return loss

    def search_parameter_gridsearch(self, train_data, kfold, param_grid = None):
        if param_grid:
            params, loss = gridsearch.run_grid_search(train_data, kfold, param_grid)
        else:
            params, loss = gridsearch.run_grid_search(train_data, kfold)
        return params, loss

    def search_parameter_randomsearch(self, train_data, kfold, iterations):
        params, loss = rs.get_parameters(train_data, kfold, iterations)
        parameter = self.format_parameter(params)

        return parameter, loss

    def search_parameter_tpe(self, train_data, kfold, iterations, save=False, filepath = './result/loss_time_tpe.csv'):
        parameter, loss = tpe.get_parameters(train_data, kfold, iterations, save=save, filepath=filepath)

        return parameter, loss

    def search_parameter_anneal(self, train_data, kfold, iterations, save=False, filepath = './result/loss_time_anneal.csv'):
        parameter, loss = anneal.get_parameters(train_data, kfold, iterations, save=save, filepath=filepath)

        return parameter, loss

    def search_parameter_bohb(self, train_data, kfold, iterations):
        params, loss = bohb.get_parameters(train_data, kfold, iterations)
        parameter = self.format_parameter(params)

        return parameter, loss

    def search_parameter_optuna(self, train_data, kfold, iterations):
        params, loss = Optuna.get_parameters(train_data, kfold, iterations)
        parameter = self.format_parameter(params)

        return parameter, loss



    def format_parameter(self, params):
        params['boosting_type'] = 'gbdt'
        params['objective'] = 'regression_l2'

        return params








