import os
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from collections import defaultdict
from catboost import CatBoostClassifier
import xgboost as xgb

from preprocessor import Preprocessor

class Trainer():
    def __init__(self, args, cfg):
        self.preprocessor = Preprocessor(args, cfg)
        self.args = args
        self.cfg = cfg

    def training(self):
        print("###start data load & preprocessing###")
        self.train, self.valid,self.test, self._train_value, self._valid_value = self.preprocessor.preprocess_total_dataset()

        print("###start MODEL training ###")
        if self.args.model == 'cat':
            self.model =  CatBoostClassifier(learning_rate= self.args.learning_rate, iterations=self.args.iterations, task_type="GPU")
            self.model.fit(self.train, self._train_value, early_stopping_rounds=100, cat_features=list(self.train.columns) ,verbose=500)

        elif self.args.model == 'xg':
            self.model = xgb.XGBClassifier(learning_rate=self.args.learning_rate,n_estimators = self.args.iterations,max_depth=9)
            self.model.fit(self.train, self._train_value ,verbose=500,early_stopping_rounds=100, eval_metric='auc',eval_set=[(self.valid, self._valid_value)])

        

    def inference(self):
        

        # submission 제출하기 위한 코드
        print("### Inference && Save###")

        _test_pred = self.model.predict_proba(self.test)[:,1]
        self.test['prediction'] = _test_pred
        submission = self.test['prediction'].reset_index(drop = True).reset_index()
        submission.rename(columns = {'index':'id'}, inplace = True)
        submission.to_csv(os.path.join(self.args.output_path, 'submission.csv'), index = False)


