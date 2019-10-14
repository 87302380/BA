import lightgbm as lgb
from sklearn.model_selection import KFold
import time

def cv_method(parameters, train_data, kfold, start_time = None):

    eval_hist = lgb.cv(parameters, train_data,
                       folds=KFold(kfold),
                       stratified=False,
                       shuffle=False,
                       verbose_eval=-1,
                       early_stopping_rounds=10)

    loss = min(eval_hist['l2-mean'])

    print(loss)
    if start_time:
        return loss, time.time()-start_time
    else:
        return loss
