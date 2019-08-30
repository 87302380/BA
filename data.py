import pandas as pd
import numpy as np
import lightgbm as lgb

class data:
    def __init__(self, file_path):

        self.data =  pd.read_csv(file_path, sep=',')
        self.values = self._get_data()
        self.label = self._get_label()
        self.feature_name = self._get_feature_name()

    def set_target_feature_index(self, target_feature_index):
        self.target_feature_index = target_feature_index


    def _get_data(self):
        data = self.data
        value = data.iloc[0:data.shape[0], 1:data.shape[1]]

        return  np.array(value)

    def _get_label(self):
        data = self.data
        lable = data['label']

        return np.array(lable)

    def _get_feature_name(self):
        data = self.data
        feature_name = list(data.columns.values)
        del feature_name[0]
        return feature_name

    def get_lgbDataset(self, target_feature_index):

        x_train = np.delete(self.values, target_feature_index, 1)
        y_train = self.values[:, target_feature_index]
        train_data = lgb.Dataset(x_train, label=y_train)

        return train_data


