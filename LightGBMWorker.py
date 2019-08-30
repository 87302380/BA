import object_function
from hpbandster.core.worker import Worker
import ConfigSpace as CS
# import lightgbm as lgb
# from sklearn.model_selection import KFold, LeaveOneOut
import time

class LightGBMWorker(Worker):

    def __init__(self, train_data, kFold, **kwargs):
        super().__init__(**kwargs)

        self.min_loss = 99
        self.train_loader = train_data
        self.kFold = kFold


    def compute(self, config, budget, *args, **kwargs):
        # max_depth = int(config['max_depth'])
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

        loss = object_function.cv_method(parameters, self.train_loader, self.kFold)

        return ({
            'loss': loss,  # this is the a mandatory field to run hyperband
            'info': loss  # can be used for any user-defined information - also mandatory
        })


    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        # config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('max_depth', lower=3, upper=5))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('num_leaves', lower=10, upper=35))
        #config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('num_trees', lower=100, upper=500))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('min_data_in_leaf', lower=1, upper=12))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('max_bin', lower=20, upper=255))
        # config_space.add_hyperparameter(CS.UniformFloatHyperparameter('bagging_fraction', lower=0.1, upper=1))
        # config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('bagging_freq', lower=0, upper=500))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('feature_fraction', lower=0.01, upper=1.0))
        # config_space.add_hyperparameter(CS.UniformFloatHyperparameter('lambda_l1', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('lambda_l2', lower=0, upper=70))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('min_gain_to_split', lower=0.0, upper=1.0))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('learning_rate', lower=0.005, upper=0.5))

        return (config_space)