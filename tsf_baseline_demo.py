import pandas as pd

import pickle
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(2018)
print("load_data")

data_feature=pd.read_csv("data/data_all_new.csv",header=0,names=["work_time","engine_speed","pump_speed","pump_pressure",
                                                             "temperature","flow","pressure","output_current","low_on",
                                                            "high_on","signal","positive_pump","anti_pump","unit_type","sample_file_name"])
print(data_feature.shape)


df_train=pd.read_csv("data/train_labels.csv")
df_test=pd.read_csv("data/submit_example.csv")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(data_feature["unit_type"])
data_feature["unit_type"]=le.transform(data_feature["unit_type"])


# features_col=["engine_speed","pump_speed","pump_pressure","temperature","flow","pressure","output_current"]
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# data_feature[features_col] = scaler.fit_transform(data_feature[features_col])

#define function for aggregation
def create_new_columns(name,aggs):
    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]

#众数   scipy.stats.modes
aggs = {
    # "sample_file_name": ["size"],
    # "work_time": ["max"],
    # "engine_speed":['max','min','mean','var',np.ptp],
    # "pump_speed":['max','min','mean','var',np.ptp],
    # "pump_pressure":['max','min','mean','var',np.ptp],
    # "temperature":['max','min','mean','var',np.ptp],
    # "flow":['max','min','mean','var','median',np.ptp],
    # "pressure":['max','min','mean','var',np.ptp],
    # "output_current":['max','min','mean','var',np.ptp],
    #
    # "low_on":["mean"],
    # "high_on":["mean"],
    # "signal":["sum",'max','var','std'],
    # "positive_pump":['sum',"mean",'var','std'],
    # "anti_pump":['sum',"max",'mean'],
    # "unit_type":["mean",],

    "sample_file_name":["size"],
    "work_time":["max"],
    "engine_speed":["mean","max","min",],
    "pump_speed":["mean","max","min"],
    "pump_pressure":["mean","max","min"],
    "temperature":["mean","max","min",np.ptp],
    "flow":["sum","mean","max","min",'var'],
    "pressure":["mean","max","min",'var'],
    "output_current":["mean","max","min",'var'],

    "low_on":["mean"],
    "high_on":["mean"],
    "signal":["sum",'max','mean'],
    "positive_pump":['sum',"mean"],
    "anti_pump":['sum',"max",'mean'],
    "unit_type":["mean",],

}

def make_feature(data,aggs,name):

    agg_df = data.groupby('sample_file_name').agg(aggs)
    agg_df.columns = agg_df.columns = ['_'.join(col).strip()+name for col in agg_df.columns.values]
    agg_df.reset_index(drop=False, inplace=True)

    agg_df["f0"]=agg_df["engine_speed_mean_tsf"]-agg_df["pump_speed_mean_tsf"]
    agg_df["f1"] = agg_df["work_time_max" + name] / agg_df["sample_file_name_size" + name]
    agg_df["f2"] = agg_df["positive_pump_sum" + name] * agg_df["sample_file_name_size" + name]
    agg_df["f5"] = agg_df["work_time_max" + name] * agg_df["sample_file_name_size" + name]


    agg_df["f6"] = (agg_df["positive_pump_sum" + name]+agg_df["anti_pump_sum" + name])/agg_df["sample_file_name_size" + name]
    agg_df["f7"]=agg_df["f6"]*agg_df["work_time_max" + name]
    return agg_df



agg_df=make_feature(data_feature,aggs,"_tsf")
df_train = df_train.merge(agg_df,on='sample_file_name',how='left')
df_test = df_test.merge(agg_df,on='sample_file_name',how='left')
#########################################################################


data_feature_a=data_feature[data_feature["positive_pump"]==1]

def make_feature_a(data,name):
    aggs = {
        #"work_time": ["count"],
        "engine_speed": ["min"],
        "pump_speed": [ "min"],
        "pump_pressure": ["min",'mean'],
        "temperature": ["min",'mean'],
        "flow": ["min",'mean'],
        "pressure": ["min",'mean'],
        "output_current": ["min"], }

    agg_df = data.groupby('sample_file_name').agg(aggs)
    agg_df.columns = agg_df.columns = ['_'.join(col).strip() + name for col in agg_df.columns.values]
    agg_df.reset_index(drop=False, inplace=True)
    return agg_df

agg_df=make_feature_a(data_feature_a,"_tsf_a")
df_train = df_train.merge(agg_df,on='sample_file_name',how='left')
df_test = df_test.merge(agg_df,on='sample_file_name',how='left')

df_test["label"]=5
data=pd.concat([df_train,df_test],axis=0, ignore_index=True)
####add_feature
df_train = data[data.label <5].copy()
df_test = data[data.label ==5].copy()
print(df_train.shape,df_test.shape)


print(df_train.shape,df_test.shape)
del data_feature
gc.collect()

df_train_columns = [c for c in df_train.columns if c not in ['sample_file_name', 'label']]
target = df_train['label']
cate_feature=["unit_type_mean_tsf","anti_pump_max_tsf","signal_max_tsf"]

print(len(df_train_columns))
print(df_train_columns)

from sklearn.metrics import f1_score

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True

param = {'num_leaves': 256,
         #'min_data_in_leaf': 50,
         'objective': 'binary',
         'max_depth': -1,
         'learning_rate': 0.05,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.95,
         "bagging_seed": 11,
         #"max_bin": 125,
         "metric": 'auc',
         #"lambda_l1": 0.5,
         #"lambda_l2": 0.01,
         "verbosity": -1,
         "nthread": 16,
         "random_state": 4038
         }
#https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4038)
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()
clf_name="tsf"
cv_scores=[]
cv_rounds=[]
for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train, target.values)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(df_train.iloc[trn_idx][df_train_columns],
                           label=target.iloc[trn_idx])  # , categorical_feature=categorical_feats)
    val_data = lgb.Dataset(df_train.iloc[val_idx][df_train_columns],
                           label=target.iloc[val_idx])  # , categorical_feature=categorical_feats)

    num_round = 20000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[val_data], verbose_eval=100,#learning_rates=lambda iter: 0.05 * (0.99 ** iter),
                    early_stopping_rounds=400,)#feval=lgb_f1_score)
    oof[val_idx] = clf.predict(df_train.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)

    cv_scores.append(f1_score(target.iloc[val_idx], oof[val_idx] > 0.5))
    cv_rounds.append(clf.best_iteration)
    print("%s now score is:" % clf_name, cv_scores)
    print("%s now round is:" % clf_name, cv_rounds)
    predictions += clf.predict(df_test[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits
score_all=f1_score(target, oof > 0.5)
print(score_all)

sub_df = pd.DataFrame({"sample_file_name":df_test["sample_file_name"].values})
sub_df["label"] = [np.round(x) for x in predictions]
sub_df["label"]=sub_df["label"].astype("int")
sub_df.to_csv("sub/classify_sub_%s_.csv"%score_all, index=False)


