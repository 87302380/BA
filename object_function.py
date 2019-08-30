import lightgbm as lgb
from sklearn.model_selection import KFold


def cv_method(parameters, train_data, kfold):
    eval_hist = lgb.cv(parameters, train_data,
                       folds=KFold(kfold),
                       stratified=False,
                       shuffle=False,
                       verbose_eval=False,
                       early_stopping_rounds=10)

    loss = min(eval_hist['l2-mean'])

    return loss
