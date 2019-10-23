This folder contains the code used to generate the illustration.

plot_gs_process.py is used to generate an illustration of a Gaussian process.

use time_fix.py to correct the result timepoint.

use ploy.py to generate illustration.(Illustration of the third chapter)

The results of all experiments in the result folder.
all.csv contains all the corrected and processed data.
all_fix.csv is the result of filtering some dirty data
best_result_all.csv is the merge of the best result of all method.
best_result_2.csv contains the best results for bohb and optuna.

The other folders are the result of the method corresponding to the folder name. For example, bohb1, bohb2, and bohb3 are different results. The first file in each folder is the merge of all the results in that folder (loss_time_bohb.csv),  A file with a fix suffix is a merge of the corrected results (loss_time_bohb_fix.csv)


