import os
import argparse

import torch
import pandas as pd
from utils import seed_everything
from trainer import Trainer
from preprocessor import Preprocessor
from cat_boost import Cat_boost
from xg_boost import Xg_boost
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
    parser.add_argument("--saved_path", type = str, default = "/opt/ml/input/level2_dkt_recsys-level2-recsys-04/boosting/weight")
    parser.add_argument("--output_path", type = str, default = "/opt/ml/input/level2_dkt_recsys-level2-recsys-04/output")
    parser.add_argument('--model', type=str, choices=['cat','xg'], default='cat',
                                help='학습 및 예측할 모델을 선택할 수 있습니다.')
    parser.add_argument("--seed", type=int, default = 42, help="seed")

    parser.add_argument('--iterations', type = int, default = 1000)
    parser.add_argument('--learning_rate', type = float, default = 3e-4)
    parser.add_argument('--check_epoch', type = int, default = 1)
    #### xg 
    parser.add_argument('--max_depth', type = int, default = 9)


    parser.add_argument("--test_size", type=float, default = 0.2, help="test set ratio")

    return parser.parse_args()


def main(args):

    ######################## DATA LOAD
    #Traininer init-> preprocessor에서 전처리 하면서 데이터 로드
    print("LOAD DATA")
    data = pd.read_csv(args.data_path + '/FE/FE_total.csv')    

    ######################## MODEL INIT
    if args.model == 'cat':
        model = Cat_boost(args)
    elif args.model == 'xg':
        model = Xg_boost(args)


        
    ######################## SELECT FEATURE
        
    FEATURE = [
        'userID', 
        'assessmentItemID', 
    #    'testId', 
    #    'answerCode', 
    #    'Timestamp',
        'KnowledgeTag', 
        'solve_time', 
        'b_category', 
        'test_category',
        'problem_id', 
        'category_st_qcut_5', 
        'last_answerCode', 
        'year', 
        'month',
        'day', 
        'hour', 
    #    'user_correct_answer',     
    #    'user_total_answer', 
    #    'user_acc',
    #    'test_mean', 
    #    'test_sum', 
    #    'tag_mean', 
    #    'tag_sum', 
    #   'user_acc_5',
    #   'tag_mean_5', 
    #   'test_mean_5'
    ]

    ######################## DATA PREPROCESSING
        
    model.preprocess(data,FEATURE)
    
        

    
    ######################## TRAIN
    
    model.training(FEATURE)

    model.inference(FEATURE)
        
    #trainer = Trainer(args,cfg)
    ######################## TRAIN
    #trainer.training()
    ######################## INFERENCE
    #trainer.inference()


    
if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    main(args)