from hyperopt import hp
from hyperopt import fmin, tpe, space_eval, Trials
from hyperopt.pyll.base import scope
import object_function
from data import data


def get_parameters(train_data, kFold, iterations):

    def objective(parameters):

        loss = object_function.cv_method(parameters, train_data, kFold)

        return loss

    configspace = {
        'boosting_type': hp.choice('boosting_type',['gbdt']),
        'objective': hp.choice('objective', ['regression_l2']),
        # hp.quniform('max_depth', -10, 10, 1),
        'num_leaves' : scope.int(hp.quniform('num_leaves',10, 35, 1)),
        'min_data_in_leaf': scope.int(hp.quniform('min_data_in_leaf', 1, 12, 1)),
        'max_bin': scope.int(hp.quniform('max_bin', 20, 255, 10)),
        'lambda_l2': scope.int(hp.quniform('lambda_l2', 0, 70, 5)),
        'feature_fraction': hp.uniform('feature_fraction', 0.01, 1),
        'min_gain_to_split': hp.uniform('min_gain_to_split', 0.0, 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.005, 0.5)

    }

    trials = Trials()

    best = fmin(objective, configspace, algo=tpe.suggest, max_evals=iterations, trials=trials)

    best_parameters = space_eval(configspace, best)
    best_loss = trials.best_trial['result']['loss']

    return best_parameters, best_loss

