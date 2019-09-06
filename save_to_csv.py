import pandas as pd
import numpy as np

def save(filepath, timepoint, loss):

    iteration = np.arange(1, len(loss)+1)
    data = pd.DataFrame({'loss':loss, 'timepoint':timepoint, 'iteration':iteration})

    data.to_csv(filepath, index=True, sep=',')

