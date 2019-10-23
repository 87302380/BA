from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from sklearn.model_selection import KFold
import object_function

pg = {
    'boosting_type': ['gbdt'],
    'objective': ['regression_l2'],
    'num_leaves': [10, 31, 64, 127],
    'max_bin': [20, 50, 100, 150, 200],
    'min_data_in_leaf': [1, 2, 4, 8, 10],
    'learning_rate': [0.05, 0.1],
}


def run_grid_search(train_data, kfold, param_grid = pg):

    kfolds = KFold(kfold)

    lgb_estimator = lgb.LGBMRegressor(learning_rate=0.01, metric='l2')
    gsearch = GridSearchCV(estimator=lgb_estimator, param_grid= param_grid, cv=kfolds)

    lgb_model = gsearch.fit(X=train_data.data, y=train_data.label)

    loss = object_function.cv_method(lgb_model.best_params_, train_data, kfold)

    return lgb_model.best_params_, loss

