import optuna
import object_function
import time
import save_to_csv

def get_parameters(train_data, kFold, iterations, save=False, filepath = './result/loss_time_optuna.csv'):

    def objective(trial):


        num_leaves = trial.suggest_int('num_leaves', 10, 35)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 1, 12)
        max_bin = trial.suggest_int('max_bin', 20, 255)
        feature_fraction = trial.suggest_uniform('feature_fraction', 0.01, 1.0)
        lambda_l2 = trial.suggest_uniform('lambda_l2', 0, 70.0)
        min_gain_to_split = trial.suggest_uniform('min_gain_to_split', 0.0, 1.0)
        learning_rate = trial.suggest_uniform('learning_rate', 0.005, 0.5)

        parameters = {
            'boosting_type': 'gbdt',
            'objective': 'regression_l2',
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            # 'max_depth': max_depth,
            'min_data_in_leaf': min_data_in_leaf,
            # 'num_trees': 10000,
            'max_bin': max_bin,
            # 'bagging_fraction': bagging_fraction,
            # 'bagging_freq': bagging_freq,
            'feature_fraction': feature_fraction,
            'verbose': -1,
            # 'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'min_gain_to_split': min_gain_to_split
        }
        if save:
            loss, timepoint = object_function.cv_method(parameters, train_data, kFold, start)
            timepoint_dic.append(timepoint)
            loss_dic.append(loss)
        else:
            loss = object_function.cv_method(parameters, train_data, kFold)

        return loss

    if save:
        start = time.time()
        timepoint_dic = []
        loss_dic = []
        study = optuna.create_study()
        study.optimize(objective, n_trials=iterations)
        save_to_csv.save(filepath, timepoint_dic, loss_dic)
    else:
        study = optuna.create_study()
        study.optimize(objective, n_trials=iterations)




    return study.best_params, study.best_value