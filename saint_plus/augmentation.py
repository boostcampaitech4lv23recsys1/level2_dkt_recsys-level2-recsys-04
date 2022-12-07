from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from config import Config
from sklearn.model_selection import train_test_split
import gc
def data_argument(train):
    _train = train.copy()
    _train.reset_index(drop = True, inplace= True)
    _train.loc[_train.drop_duplicates(subset='userID', keep = 'last').index, 'tem'] = -1
    _valid = _train[_train['tem'] == -1]
    _train = _train[_train['tem'] == 0]
    return _train, _valid


def get_dataloaders():
    dtypes = {
        'userID': 'int16',
        'answerCode': 'int8',
        'KnowledgeTag': 'int16',
    }
    print("loading csv.....")
    dat = pd.read_csv("/opt/ml/input/data/FE_total.csv", usecols=[0, 1, 3, 4, 5], dtype=dtypes, parse_dates=['Timestamp'])  # TestID 빼고 볼러옴

    dat = dat.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
    dat['tem'] = 0
    _train = dat[dat['answerCode'] >= 0]
    _test = dat[dat['answerCode'] == -1]

    _train_x, _valid = data_argument(_train)
    arg_train = []
    now = _train_x
    for _ in range(Config.AUGMENTATION):
        now, now1 = data_argument(now)
        arg_train.append(data_merge(now, now1))
    test = data_merge(_train, _test)
    valid = data_merge(_train_x, _valid)
    train = pd.concat(arg_train)

    
     #data augmentation
    if Config.DATA_AUG:
        train_origin = train_df.copy()
        n= 1
        print(f'======origin length      : {len(train_df)}======')
        for i in range(Config.AUGMENTATION):
            print(f'START {n}th AUGMENTATION')
            tem = train_origin.duplicated(subset = ["userID"], keep = "last")
            train_origin = train_origin.drop(index=tem.index)
            train_origin['userID'] += train_origin['userID'].nunique()
            train = pd.concat([train_df, train_origin], axis = 0)
            print(f'END   {n}th AUGMENTATION')
            n += 1
        print(f'======after augmentation : {len(train_df)}======')

    train_df.to_csv("/opt/ml/input/data/train.csv", index=True)
get_dataloaders()