import logging
import lightgbm as lgb
import numpy as np
from data import data
from sklearn.model_selection import KFold

import ConfigSpace as CS

from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO

def get_parameters(train_data, kfold, iterations):

    def compute(config):

        num_leaves = int(config['num_leaves'])
        max_bin = int(config['max_bin'])
        min_data_in_leaf = int(config['min_data_in_leaf'])
        # num_trees = int(config['num_trees'])
        # bagging_fraction = config['bagging_fraction']
        # bagging_freq = int(config['bagging_freq'])
        feature_fraction = config['feature_fraction']
        # lambda_l1 = config['lambda_l1'],
        lambda_l2 = config['lambda_l2'],
        min_gain_to_split = config['min_gain_to_split']
        learning_rate = config['learning_rate']

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

        eval_hist = lgb.cv(parameters, train_data,
                           folds=KFold(kfold),
                           stratified=False,
                           shuffle=False,
                           verbose_eval=True,
                           early_stopping_rounds=10)


        loss = min(eval_hist['l2-mean'])


        # loss = parameters['num_leaves'] * parameters['num_leaves']
        return loss



    logging.basicConfig(level=logging.DEBUG)  # logging.DEBUG for debug output

    # Build Configuration Space which defines all parameters and their ranges

    def get_configspace():
        config_space = CS.ConfigurationSpace()
        # config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('max_depth', lower=3, upper=5))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('num_leaves', lower=10, upper=35))
        # config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('num_trees', lower=100, upper=500))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('min_data_in_leaf', lower=1, upper=12))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('max_bin', lower=20, upper=255))
        # config_space.add_hyperparameter(CS.UniformFloatHyperparameter('bagging_fraction', lower=0.1, upper=1))
        # config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('bagging_freq', lower=0, upper=500))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('feature_fraction', lower=0.01, upper=1.0))
        # config_space.add_hyperparameter(CS.UniformFloatHyperparameter('lambda_l1', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('lambda_l2', lower=0, upper=70))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('min_gain_to_split', lower=0.0, upper=1.0))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('learning_rate', lower=0.005, upper=0.5))

        return (config_space)

    cs = get_configspace()
    # Scenario object
    scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                         "runcount-limit": iterations,   # max. number of function evaluations; for this example set to a low number
                         "cs": cs,               # configuration space
                         "deterministic": "false"
                         })

    # Example call of the function
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = compute(cs.get_default_configuration())
    print("Default Value: %.2f" % def_value)

    # Optimize, using a SMAC-object
    print("Optimizing! Depending on your machine, this might take a few minutes.")
    smac = SMAC4HPO(scenario=scenario,
                   rng=np.random.RandomState(42),
                   tae_runner=compute,
                   )


    incumbent = smac.optimize()
    print("haole")

    inc_value = compute(incumbent)

    return inc_value

path = "colon_label_in_first_row.csv"
data = data(path)

train_data = data.get_lgbDataset(2)

get_parameters(train_data, 3, 5)