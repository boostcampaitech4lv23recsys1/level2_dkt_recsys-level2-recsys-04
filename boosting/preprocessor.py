import os
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

class Preprocessor():
    def __init__(self, args, cfg):
        self.data_path = args.data_path
        self.saved_path = args.saved_path
        self.output_path = args.output_path
        self.cfg = cfg
        self.args = args
        pd.set_option('mode.chained_assignment',  None)


    def xg_preprocess(self):
        ##### label encodig
        dat = self._load_total_dataset()

       
        ##### label encodig
        encoder = LabelEncoder()
        encoder.fit(dat['assessmentItemID'])
        dat['assessmentItemID'] = encoder.transform(dat['assessmentItemID'])

        encoder.fit(dat['category_st_qcut_5'])
        dat['category_st_qcut_5'] = encoder.transform(dat['category_st_qcut_5'])

        encoder.fit(dat['testId'])
        dat['testId'] = encoder.transform(dat['testId'])

        self._train = dat[dat['answerCode'] >= 0]
        self._test = dat[dat['answerCode'] < 0]

        self._split_train_valid_dataset()

        # 모델에 적용하기 전 기본적인 데이터 전처리 부분
        ## 라벨링, 필요없는 칼럼 제거
        self._train_value = self._train['answerCode']
        self._train.drop(['Timestamp', 'train_valid', 'answerCode'], axis = 1, inplace = True)

        self._valid_value = self._valid['answerCode']
        self._valid.drop(['Timestamp', 'train_valid', 'answerCode'], axis = 1, inplace = True)

        self._test.drop(['Timestamp', 'answerCode'], axis = 1, inplace = True)



    def cat_preprocess(self):
        # 모델에 적용하기 전 기본적인 데이터 전처리 부분
        ## 라벨링, 필요없는 칼럼 제거
        dat = self._load_total_dataset()

        self._train = dat[dat['answerCode'] >= 0]
        self._test = dat[dat['answerCode'] < 0]

        self._split_train_valid_dataset()

        self._train_value = self._train['answerCode']
        self._train.drop(['Timestamp', 'testId', 'train_valid', 'answerCode'], axis = 1, inplace = True)

        self._valid_value = self._valid['answerCode']
        self._valid.drop(['Timestamp', 'testId', 'train_valid', 'answerCode'], axis = 1, inplace = True)

        self._test.drop(['Timestamp', 'testId', 'answerCode'], axis = 1, inplace = True)


        # CatBoost에 적용하기 위해선 문자열 데이터로 변환 필요.
        self._train['userID'] = self._train['userID'].astype('str')
        self._train['KnowledgeTag'] = self._train['KnowledgeTag'].astype('str')

        self._valid['userID'] = self._valid['userID'].astype('str')
        self._valid['KnowledgeTag'] = self._valid['KnowledgeTag'].astype('str')

        self._test['userID'] = self._test['userID'].astype('str')
        self._test['KnowledgeTag'] = self._test['KnowledgeTag'].astype('str')


    def _load_total_dataset(self):
        dat = pd.read_csv(os.path.join(self.data_path, 'FE/FE_total.csv'))
        return dat

    def _split_train_valid_dataset(self):
        user_final_time = self._train.groupby('userID')['Timestamp'].max()
        self._train['train_valid'] = self._train.apply(lambda x : -1 if x.Timestamp == user_final_time[x.userID] else x['answerCode'], axis = 1)
        self._valid = self._train[self._train['train_valid'] < 0]
        self._train = self._train[self._train['train_valid'] >= 0]

        
    
    def preprocess_total_dataset(self):
        
        
        if self.args.model == 'cat':
            print("###Cat Boost Preprocesses###")
            self.cat_preprocess()
        elif self.args.model == 'xg':
            print("###XG Boost Preprocesses###")
            self.xg_preprocess()
        return self._train, self._valid, self._test, self._train_value, self._valid_value