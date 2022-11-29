import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import build_dense_graph
import random
import warnings
warnings.filterwarnings(action='ignore')

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com


class KTDataset(Dataset):
    #데이터를 features / questions / answers 로 분할하여 저장(type : list)
    def __init__(self, features, questions, answers):
        super(KTDataset, self).__init__()
        self.features = features
        self.questions = questions
        self.answers = answers

    def __getitem__(self, index):
        return self.features[index], self.questions[index], self.answers[index]

    def __len__(self):
        return len(self.features)


def pad_collate(batch):
    #batch(split) -> feature_pad, question_pad, answer_pad
    (features, questions, answers) = zip(*batch)
    features = [torch.LongTensor(feat) for feat in features]
    questions = [torch.LongTensor(qt) for qt in questions]
    answers = [torch.LongTensor(ans) for ans in answers]
    feature_pad = pad_sequence(features, batch_first=True, padding_value=-1)
    question_pad = pad_sequence(questions, batch_first=True, padding_value=-1)
    answer_pad = pad_sequence(answers, batch_first=True, padding_value=-1)
    return feature_pad, question_pad, answer_pad


def load_dataset(file_path,max_seq_len_limit,test_valid_len, batch_size, graph_type, dkt_graph_path=None, train_ratio=0.9, val_ratio=0.1, shuffle=True, model_type='GKT', use_binary=True, res_len=2, use_cuda=True):
    r"""
    Parameters:
        file_path: input file path of knowledge tracing data
        batch_size: the size of a student batch
        graph_type: the type of the concept graph
        shuffle: whether to shuffle the dataset or not
        use_cuda: whether to use GPU to accelerate training speed
    Return:
        concept_num: KnowledgeTag의 고유값 개수
        graph: the static graph is graph type is in ['Dense', 'Transition', 'DKT'], otherwise graph is None; conept_num을 이용하여 만듬
        train_data_loader: data loader of the training dataset
        valid_data_loader: data loader of the validation dataset
        test_data_loader: data loader of the test dataset
    NOTE: stole some code from https://github.com/lccasagrande/Deep-Knowledge-Tracing/blob/master/deepkt/data_util.py
    """
    # read csv data
    dtype = {
        'userID': 'int16',
        'answerCode': 'int8',
        'KnowledgeTag': 'int16'
    }
    DATA_PATH = '/opt/ml/input/data/'
    train = pd.read_csv(DATA_PATH + 'train_data.csv', dtype=dtype, parse_dates=['Timestamp'])
    test = pd.read_csv(DATA_PATH + 'test_data.csv', dtype=dtype, parse_dates=['Timestamp'])
    
    
    ########## 여기에서 우리가 썼던 train test 셋으로 나누고 train validaiton 분리 시킨다.
    #test_valid_len (default ) = 3
    dat = pd.concat([train, test.drop(test.tail(test_valid_len).index)], axis = 0)
    dat = dat.sort_values(by = ['userID', 'Timestamp'])
    
    ########### 여기에서 max sequence 값을 데이터 선에서 짤라주자
    dat = dat.groupby('userID', sort=False).tail(max_seq_len_limit)


    train = dat[dat['answerCode'] >= 0]
    test = test.sort_values(by = ['userID', 'Timestamp'])
    test = test.groupby('userID', sort=False).tail(test_valid_len)

    #######train vaild split
    #user_final_time = train.groupby('userID')['Timestamp'].max()
    #train['train_valid'] = train.apply(lambda x : -1 if x.Timestamp == user_final_time[x.userID] else x['answerCode'], axis = 1)
    #valid = train[train['train_valid'] < 0]
    #train = train[train['train_valid'] >= 0]

    # if "KnowledgeTag" not in df.columns:
    #     raise KeyError(f"The column 'KnowledgeTag' was not found on {file_path}")
    # if "answerCode" not in df.columns:
    #     raise KeyError(f"The column 'answerCode' was not found on {file_path}")
    # if "userID" not in df.columns:
    #     raise KeyError(f"The column 'userID' was not found on {file_path}")

    # if not (df['answerCode'].isin([0, 1])).all():
    #     raise KeyError(f"The values of the column 'answerCode' must be 0 or 1.")

    # Step 1.1 - Remove questions without skill
    # df.dropna(subset=['KnowledgeTag'], inplace=True)  # KnowledgeTag : na인 값 없음

    # Step 1.2 - Remove users with a single answer
    # df = df.groupby('userID').filter(lambda q: len(q) > 1).copy()  # 문제 한 개만 푼 userID 없음

    # Step 2 - Enumerate skill id
    train['KTag'], _ = pd.factorize(train['KnowledgeTag'], sort=True)
    test['KTag'], _ = pd.factorize(test['KnowledgeTag'], sort=True)

    # test 데이터의 각 유저 별 마지막 문제 맞았다고 가정 -> 나중에 예측에서 진짜 맞았을 확률 토해냄 -> 이게 우리 리더보드 채점 방식에 맞는 것 같음
    # 틀렸다고 가정하면 -> 진짜 틀렸을 확률 토해낼 듯
    test.loc[test['answerCode'] == -1, 'answerCode'] = 1

    # Step 3 - Cross skill id with answer to form a synthetic feature
    # use_binary: (0,1); !use_binary: (1,2,3,4,5,6,7,8,9,10,11,12). Either way, the answerCode result index is guaranteed to be 1
    if use_binary:
        train['KTag_wiht_answer'] = train['KTag'] * 2 + train['answerCode']
        test['KTag_wiht_answer'] = test['KTag'] * 2 + test['answerCode']
    else:
        train['KTag_wiht_answer'] = train['KTag'] * res_len + train['answerCode'] - 1
        test['KTag_wiht_answer'] = test['KTag'] * res_len + test['answerCode'] - 1


    #######train vaild split
    train = train.sort_values(by = ['userID', 'Timestamp'])
    valid = train.groupby('userID', sort=False).tail(test_valid_len)
    train = train.drop(train.tail(test_valid_len).index)  

    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    train_feature_list = []
    train_question_list = []
    train_answer_list = []
    train_seq_len_list = []

    def train_get_data(series):
        train_feature_list.append(series['KTag_wiht_answer'].tolist())
        train_question_list.append(series['KTag'].tolist())
        train_answer_list.append(series['answerCode'].eq(1).astype('int').tolist())
        train_seq_len_list.append(series['answerCode'].shape[0])
    
    # 우리 대회 데이터는 이미 Train, Test 분리되었기 때문에, Test 데이터를 위한 리스트들 따로 만들고 진행함
    test_feature_list = []
    test_question_list = []
    test_answer_list = []
    test_seq_len_list = []
    
    def test_get_data(series):
        test_feature_list.append(series['KTag_wiht_answer'].tolist())
        test_question_list.append(series['KTag'].tolist())
        test_answer_list.append(series['answerCode'].eq(1).astype('int').tolist())
        test_seq_len_list.append(series['answerCode'].shape[0])

    valid_feature_list = []
    valid_question_list = []
    valid_answer_list = []
    valid_seq_len_list = []
    
    def valid_get_data(series):
        valid_feature_list.append(series['KTag_wiht_answer'].tolist())
        valid_question_list.append(series['KTag'].tolist())
        valid_answer_list.append(series['answerCode'].eq(1).astype('int').tolist())
        valid_seq_len_list.append(series['answerCode'].shape[0])
        
    train.groupby('userID').apply(train_get_data)
    test.groupby('userID').apply(test_get_data)
    valid.groupby('userID').apply(valid_get_data)

    question_list = train_question_list + test_question_list
    seq_len_list = train_seq_len_list + test_seq_len_list

    max_seq_len = np.max(seq_len_list)
    print('max seq_len: ', max_seq_len)
    student_num = len(seq_len_list)
    print('student num: ', student_num)
    feature_dim = int(train['KTag_wiht_answer'].max() + 1)  # 테스트 데이터의 ktag 어차피 다 트레인 데이터 안에 있어서, 이렇게 해도 상관 없을듯
    print('feature_dim: ', feature_dim)
    question_dim = int(train['KTag'].max() + 1)
    print('question_dim: ', question_dim)
    concept_num = question_dim

    # print('feature_dim:', feature_dim, 'res_len*question_dim:', res_len*question_dim)
    # assert feature_dim == res_len * question_dim

    train_dataset = KTDataset(train_feature_list, train_question_list, train_answer_list)
    test_dataset = KTDataset(test_feature_list, test_question_list, test_answer_list)
    val_dataset = KTDataset(valid_feature_list, valid_question_list, valid_answer_list)

    # user 별 마지막으로 푼 문제 index 저장
    test_last_q_idx = [len(q)-2 for q, _, _ in test_dataset]

    tot_size = len(train_seq_len_list) + len(valid_seq_len_list)
    train_size = len(train_seq_len_list)
    val_size = len(valid_seq_len_list)
    test_size = len(test_seq_len_list)

    #train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    print('train_size: ', train_size, 'val_size: ', val_size, 'test_size: ', test_size)
    
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    valid_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    # 나중을 위해 shuffle=False로, batch_size=test_size로 한 번에
    test_data_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False, collate_fn=pad_collate)

    graph = None
    if model_type == 'GKT':
        if graph_type == 'Dense':
            graph = build_dense_graph(concept_num)
        elif graph_type == 'Transition':
            graph = build_transition_graph(question_list, seq_len_list, train_dataset.indices, student_num, concept_num)
        elif graph_type == 'DKT':
            graph = build_dkt_graph(dkt_graph_path, concept_num)
        if use_cuda and graph_type in ['Dense', 'Transition', 'DKT']:
            graph = graph.cuda()
    
    return concept_num, graph, train_data_loader, valid_data_loader, test_data_loader, test_last_q_idx



def build_transition_graph(question_list, seq_len_list, indices, student_num, concept_num):
    graph = np.zeros((concept_num, concept_num))
    student_dict = dict(zip(indices, np.arange(student_num)))
    for i in range(student_num):
        if i not in student_dict:
            continue
        questions = question_list[i]
        seq_len = seq_len_list[i]
        for j in range(seq_len - 1):
            pre = questions[j]
            next = questions[j + 1]
            graph[pre, next] += 1
    np.fill_diagonal(graph, 0)
    # row normalization
    rowsum = np.array(graph.sum(1))
    def inv(x):
        if x == 0:
            return x
        return 1. / x
    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = np.diag(r_inv)
    graph = r_mat_inv.dot(graph)
    # covert to tensor
    graph = torch.from_numpy(graph).float()
    return graph


def build_dkt_graph(file_path, concept_num):
    graph = np.loadtxt(file_path)
    assert graph.shape[0] == concept_num and graph.shape[1] == concept_num
    graph = torch.from_numpy(graph).float()
    return graph