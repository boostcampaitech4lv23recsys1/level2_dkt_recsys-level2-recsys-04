import os
import argparse
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import pandas as pd
from cat_boost import Cat_boost
import xgboost as xgb
from xg_boost import Xg_boost
import numpy as np
from sklearn.model_selection import StratifiedKFold
# from lightgbm import lightgbm
import warnings
warnings.filterwarnings(action='ignore')

class cfg: 
    gpu_idx = 0
    device = torch.device("cuda:{}".format(gpu_idx) if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(
        description="[RECCAR] DKT"
    )

    ############### BASIC OPTION
    parser.add_argument("--data_path", type = str, default = "/opt/ml/input/data/")
    parser.add_argument("--saved_path", type = str, default = "/opt/ml/input/code/level2_dkt_recsys-level2-recsys-04/boosting/weight")
    parser.add_argument("--output_path", type = str, default = "/opt/ml/input/code/level2_dkt_recsys-level2-recsys-04/output")
    parser.add_argument('--model', type=str, choices=['cat','xg'], default='cat',
                                help='학습 및 예측할 모델을 선택할 수 있습니다.')
    parser.add_argument("--seed", type=int, default = 42, help="seed")

    parser.add_argument('--iterations', type = int, default = 1000)
    parser.add_argument('--learning_rate', type = float, default = 3e-4)
    parser.add_argument('--check_epoch', type = int, default = 1)
    #### xg 
    parser.add_argument('--max_depth', type = int, default = 9)

    parser.add_argument("--test_size", type=float, default = 0.2, help="test set ratio")
    parser.add_argument("--argument_times", type=int, default = 10) #데이터 증강 횟수 지정

    return parser.parse_args()

def objective(trial, FEATURE, train, valid, test, train_value, valid_value):
    
    if args.model == "cat":
        param = {
        'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.5),
        "iterations" : trial.suggest_int("iterations", 500, 1000),
        "task_type" : "GPU"
        }
        
        model = CatBoostClassifier(**param, devices = '0')
        model.fit(train[FEATURE], train_value, early_stopping_rounds=100, cat_features=list(train[FEATURE]) ,verbose=500)
        
        print('train score')
        train_pred = model.predict_proba(train[FEATURE])[:,1]
        print(roc_auc_score(train_value, train_pred)) # auc
        print(accuracy_score(train_value, np.where(train_pred >= 0.5, 1, 0))) # acc, 정확도

        print('valid score')
        valid_pred = model.predict_proba(valid[FEATURE])[:,1]
        print(roc_auc_score(valid_value, valid_pred)) # auc
        print(accuracy_score(valid_value, np.where(valid_pred >= 0.5, 1, 0))) # acc, 정확도
        
    else:
        param = {
        'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.5),
        "n_estimators" : trial.suggest_int("iterations", 500, 1000),
        "max_depth" : trial.suggest_int("max_depth", 5, 9)
        }
        model = xgb.XGBClassifier(**param)
        model.fit(train[FEATURE], train_value, verbose=100,early_stopping_rounds=100, eval_metric='auc',eval_set=[(valid[FEATURE], valid_value)])
    
        print('train score')
        train_pred = model.predict_proba(train[FEATURE])[:,1]
        print(roc_auc_score(train_value, train_pred)) # auc
        print(accuracy_score(train_value, np.where(train_pred >= 0.5, 1, 0))) # acc, 정확도

        print('valid score')
        valid_pred = model.predict_proba(valid[FEATURE])[:,1]
        print(roc_auc_score(valid_value, valid_pred)) # auc
        print(accuracy_score(valid_value, np.where(valid_pred >= 0.5, 1, 0))) # acc, 정확도


    return roc_auc_score(valid_value, valid_pred)

def main(args):

    ######################## DATA LOAD
    #Traininer init-> preprocessor에서 전처리 하면서 데이터 로드
    print("LOAD DATA")
    data = pd.read_csv(args.data_path + '/FE_total2.csv')   
    ######################## MODEL INIT
    if args.model == 'cat':
        model = Cat_boost(args)
    elif args.model == 'xg':
        model = Xg_boost(args)
        
    ######################## SELECT FEATURE
        
    FEATURE = [
        'last_answerCode', 
        'last_answerCode2', 
        'last_answerCode3', 
        'last_answerCode4', 
        'last_answerCode5', 
        'last_answerCode6', 
        'last_answerCode7', 
        'last_answerCode8', 
        'last_answerCode9', 
        'last_answerCode10', 
        'answer_mean', 
        'answer_cnt', 
        'time_mean', 
        'tag_mode', 
        'user_item_mean', 
        'item_mean', 
        'item_sum', 
        'item_time_mean'
    ]        

    ######################## DATA PREPROCESSING
    
    print("DATA PREPROCESSING")
    train, valid, test = model.preprocess(data,FEATURE)           
    train_value, valid_value = train['answerCode'], valid['answerCode']
    train.drop(['answerCode'],axis=1)
    valid.drop(['answerCode'],axis=1)
    ######################## TRAIN
    print("TRAIN")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name = f'{args.model}_parameter_opt',
        direction = 'maximize',
        sampler = sampler,
    )
    study.optimize(lambda trial : objective(trial, FEATURE, train, valid, test, train_value, valid_value), n_trials=3)
    
    ######################## BEST TRAIN & INFERENCE k-fold 미적용 version
    # if args.model == "cat":
    #     model = CatBoostClassifier(**study.best_params, devices = '0')
    #     model.fit(train[FEATURE], train_value, early_stopping_rounds=100, cat_features=list(train[FEATURE]) ,verbose=500)
    # else:
    #     model = xgb.XGBClassifier(**study.best_params)
    #     model.fit(train[FEATURE], train_value, verbose=100,early_stopping_rounds=100, eval_metric='auc',eval_set=[(valid[FEATURE], valid_value)])
    
    # test_pred = model.predict_proba(test[FEATURE])[:,1]
    # test['prediction'] = test_pred
    # submission = test['prediction'].reset_index(drop = True).reset_index()
    # submission.rename(columns = {'index':'id'}, inplace = True)
    # submission.to_csv(os.path.join(args.output_path, 'submission.csv'), index = False)
    
    ######################## BEST TRAIN & INFERENCE
    test['prediction'] = 0
    skf = StratifiedKFold(n_splits=10)
    
    for train_idx, value_idx in skf.split(train, train_value):   
        _train = train.iloc[train_idx, :]     
        _train_value = train_value.iloc[train_idx]        
        
        if args.model == "cat":
            model = CatBoostClassifier(**study.best_params, devices = '0')
            model.fit(_train[FEATURE], _train_value, early_stopping_rounds=100, cat_features=list(train[FEATURE]) ,verbose=500)            
        else:
            model = xgb.XGBClassifier(**study.best_params)
            model.fit(_train[FEATURE], _train_value, verbose=100,early_stopping_rounds=100, eval_metric='auc',eval_set=[(valid[FEATURE], valid_value)])
        test_pred = model.predict_proba(test[FEATURE])[:,1]
        test['prediction'] += test_pred
        
        print(f'================================================================================\n\n')
        
    test['prediction'] /= 10
    submission = test['prediction'].reset_index(drop = True).reset_index()
    submission.rename(columns = {'index':'id'}, inplace = True)
    submission.to_csv(os.path.join(args.output_path, 'submission.csv'), index = False)
    print("SAVE COMPLETE")
if __name__ == '__main__':
    args = parse_args()
    main(args)