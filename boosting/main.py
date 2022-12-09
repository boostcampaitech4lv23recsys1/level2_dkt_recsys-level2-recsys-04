import os
import argparse

import torch
import pandas as pd
from cat_boost import Cat_boost
from xg_boost import Xg_boost
from lightgbm import lightgbm
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
    
    model.preprocess(data,FEATURE)           
    
    ######################## TRAIN
    
    model.training(FEATURE)

    model.inference(FEATURE)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)