from bayes_opt import BayesianOptimization
import time
import object_function

def get_parameters(train_data, kFold, iterations, save=False, filepath = './result/loss_time_gs.csv'):
    def func(parameters):

        if save:
            loss, timepoint = object_function.cv_method(parameters, train_data, kFold, start)
            timepoint_dic.append(timepoint)
            loss_dic.append(loss)
        else:
            loss = object_function.cv_method(parameters, train_data, kFold)
        return -loss

    def black_box_function(num_leaves, max_bin, min_data_in_leaf, feature_fraction, lambda_l2, min_gain_to_split, learning_rate):

        parameters = {
            'boosting_type': 'gbdt',
            'objective': 'regression_l2',
            'learning_rate': learning_rate,
            'num_leaves': int(num_leaves),
            # 'max_depth': max_depth,
            'min_data_in_leaf': int(min_data_in_leaf),
            # 'num_trees': 10000,
            'max_bin': int(max_bin),
            # 'bagging_fraction': bagging_fraction,
            # 'bagging_freq': bagging_freq,
            'feature_fraction': feature_fraction,
            'verbose': -1,
            # 'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'min_gain_to_split': min_gain_to_split
        }

        return func(parameters)

    configspace = {
        'num_leaves': (10, 35),
        'min_data_in_leaf': (1,12),
        'max_bin': (20, 255),
        'feature_fraction': (0.01, 1.0),
        'lambda_l2': (0, 70),
        'min_gain_to_split': (0, 10),
        'learning_rate': (0.005, 0.5)
    }


    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=configspace,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    if save:
        start = time.time()
        timepoint_dic = []
        loss_dic = []
    else:
        optimizer.maximize(init_points=5, n_iter=iterations - 5)

        loss = -optimizer.max['target']
        params = optimizer.max['params']

    return params, loss