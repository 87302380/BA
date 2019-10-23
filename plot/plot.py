import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="darkgrid")


data = pd.read_csv('./result/best_result_2.csv')
data = data.loc[data['tuna'].isin(['optuna'])]

sns.lineplot(x="timepoint", y="loss", hue='tuna', data=data)

plt.show()