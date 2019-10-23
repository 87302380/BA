import  pandas as pd

# In order to make the drawing clearer, correct the time and round off

name = 'gs'
filepath = './result/'+ name+'/loss_time_' + name  + '.csv'
data = pd.read_csv(filepath)

data['timepoint'] = round(data['timepoint'])


print()
for i in  range(0,data['timepoint'].shape[0]):

    y = data['timepoint'][i] % 5
    x = data['timepoint'][i] // 5
    if y > 2 :
        data['timepoint'][i] = (x+1) * 5

    else:
        data['timepoint'][i] = x * 5

filepatht = './result/'+ name+'/loss_time_' + name  + '_fix.csv'
data.to_csv(filepatht, index=False)
# print(bohb)