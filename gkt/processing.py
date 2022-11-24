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
    (features, questions, answers) = zip(*batch)
    features = [torch.LongTensor(feat) for feat in features]
    questions = [torch.LongTensor(qt) for qt in questions]
    answers = [torch.LongTensor(ans) for ans in answers]
    feature_pad = pad_sequence(features, batch_first=True, padding_value=-1)
    question_pad = pad_sequence(questions, batch_first=True, padding_value=-1)
    answer_pad = pad_sequence(answers, batch_first=True, padding_value=-1)
    return feature_pad, question_pad, answer_pad


def load_dataset(file_path, batch_size, graph_type, dkt_graph_path=None, train_ratio=0.7, val_ratio=0.2, shuffle=True, model_type='GKT', use_binary=True, res_len=2, use_cuda=True):
    r"""
    Parameters:
        file_path: input file path of knowledge tracing data
        batch_size: the size of a student batch
        graph_type: the type of the concept graph
        shuffle: whether to shuffle the dataset or not
        use_cuda: whether to use GPU to accelerate training speed
    Return:
        concept_num: the number of all concepts(or questions) skill_id의 고유값 개수
        graph: the static graph is graph type is in ['Dense', 'Transition', 'DKT'], otherwise graph is None; conept_num을 이용하여 만든값
        train_data_loader: data loader of the training dataset
        valid_data_loader: data loader of the validation dataset
        test_data_loader: data loader of the test dataset
    NOTE: stole some code from https://github.com/lccasagrande/Deep-Knowledge-Tracing/blob/master/deepkt/data_util.py
    FIXME : 밑에 코드 계속 수정작업중이라 원본 데이터 많이 날아가있고 실행도 안 됩니다. 적절히 걸러서 읽어주세요.
    1. 데이터 불러오기
    2. 데이터 전처리(결측치, 범주화 등등)
    3. skill_with_answer, skill, correct 피쳐를 이용하여 데이터셋 만듬
    """
    df = pd.read_csv('data/FE_total.csv')
    # Step 1.1 - Remove questions without KnowledgeTag
    df.dropna(subset=['KnowledgeTag'], inplace=True)

    # Step 1.2 - Remove users with a single answer
    df = df.groupby('userID').filter(lambda q: len(q) > 1).copy()

    # Step 2 - Enumerate skill id
    df['skill'], _ = pd.factorize(df['KnowledgeTag'], sort=True)  # we can also use problem_id to represent exercises

    # Step 3 - Cross skill id with answer to form a synthetic feature
    # use_binary: (0,1); !use_binary: (1,2,3,4,5,6,7,8,9,10,11,12). Either way, the correct result index is guaranteed to be 1
    if use_binary:
        df['skill_with_answer'] = df['skill'] * 2 + df['answerCode']
    else:
        df['skill_with_answer'] = df['skill'] * res_len + df['answerCode'] - 1
    def get_data(series):
        feature_list.append(series['skill_with_answer'].tolist())
        question_list.append(series['skill'].tolist())
        answer_list.append(series['answerCode'].eq(1).astype('int').tolist())
        seq_len_list.append(series['answerCode'].shape[0])
    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    feature_list = []
    question_list = []
    answer_list = []
    seq_len_list = []
    df.groupby('userID').apply(get_data)
    question_dim = int(df['skill'].max() + 1)
    concept_num = question_dim
    _train = df[df['answerCode'] >= 0]
    _test = df[df['answerCode'] < 0]
    user_final_time = _train.groupby('userID')['Timestamp'].max()
    _train['train_valid'] = _train.apply(lambda x : -1 if x.Timestamp == user_final_time[x.userID] else x['answerCode'], axis = 1)
    _valid = _train[_train['train_valid'] < 0]
    _train = _train[_train['train_valid'] >= 0]

    df.groupby('userID').apply(get_data)
    max_seq_len = np.max(seq_len_list)
    print('max seq_len: ', max_seq_len)
    student_num = len(seq_len_list)
    print('student num: ', student_num)
    feature_dim = int(df['skill_with_answer'].max() + 1)
    print('feature_dim: ', feature_dim)
    question_dim = int(df['skill'].max() + 1)
    print('question_dim: ', question_dim)
    concept_num = question_dim

    # Step 5 - train / valid / test
    
    feature_list = []
    question_list = []
    answer_list = []
    seq_len_list = []
    _train.groupby('userID').apply(get_data)
    train_dataset = KTDataset(feature_list, question_list, answer_list)

    feature_list = []
    question_list = []
    answer_list = []
    seq_len_list = []
    _valid.groupby('userID').apply(get_data)
    val_dataset = KTDataset(feature_list, question_list, answer_list)

    feature_list = []
    question_list = []
    answer_list = []
    seq_len_list = []
    _test.groupby('userID').apply(get_data)
    test_dataset = KTDataset(feature_list, question_list, answer_list)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    valid_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)

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
    
    return concept_num, graph, train_data_loader, valid_data_loader, test_data_loader



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