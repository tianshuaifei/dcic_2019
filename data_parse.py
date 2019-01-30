import pandas as pd
import os
from tqdm import *
path="kaggle/huosaiData/"
data_list = os.listdir(path+'data_train/')


file_name='data/data_all_new.csv'
df = pd.read_csv(path+'data_train/'+ data_list[0])
df['sample_file_name'] = data_list[0]
df.to_csv(file_name, index=False)

for i in tqdm(range(1, len(data_list))):
    if data_list[i].split('.')[-1] == 'csv':
        df = pd.read_csv(path+'data_train/' + data_list[i])
        df['sample_file_name'] = data_list[i]
        df.to_csv(file_name, index=False, header=False, mode='a+')
    else:
        continue


test_data_list = os.listdir(path+'data_test/')


for i in tqdm(range(len(test_data_list))):
    if test_data_list[i].split('.')[-1] == 'csv':
        df = pd.read_csv(path+'data_test/' + test_data_list[i])
        df['sample_file_name'] = test_data_list[i]
        df.to_csv(file_name, index=False, header=False, mode='a+')
    else:
        continue
