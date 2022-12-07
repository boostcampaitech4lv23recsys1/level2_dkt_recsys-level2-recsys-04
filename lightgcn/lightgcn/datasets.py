import os

import pandas as pd
import torch
import warnings

warnings.filterwarnings(action='ignore')

def prepare_dataset(device, basepath, verbose=True, logger=None):
    """_summary_
    Returns:
        _type_: train dic, test dic, id+item 개수 길이
        dic : 
            edge : [[유저임베딩] * id+item 개수길이, [아이템임베딩] * id+item 개수길이], (2, 2475962)
            label : [라벨] * id+item 개수길이, (2475962)
    """    
    # 데이터 불러오고 train, test 합치기
    data = load_data(basepath)
    # answerCode가 -1인 값만 test로 둠.
    train_data, test_data = separate_data(data)

    # train을 train과 valid로 나눠줌(베이스라인에 없는 코드 추가.)
    train_data, valid_data = separate_valid(train_data)

    # 유저+아이템 모두 인덱싱한 결과값 배출.
    id2index = indexing_data(data)
    # edge : [[유저1, 유저2, 유저3 ....], [아이템1, 아이템2 .....]], label(정답유무) : [라벨1, 라벨2, ....]
    # 딕셔너리 형태로 배출
    train_data_proc = process_data(train_data, id2index, device, True)
    # 베이스라인에 없는 코드(valid) 추가
    valid_data_proc = process_data(valid_data, id2index, device, False)
    test_data_proc = process_data(test_data, id2index, device, False)

    # 정보 출력하는 부분.
    if verbose:
        print_data_stat(train_data, "Train", logger=logger)
        print_data_stat(test_data, "Test", logger=logger)

    return train_data_proc, valid_data_proc, test_data_proc, len(id2index)


def load_data(basepath):
    path1 = os.path.join(basepath, "train_data.csv")
    path2 = os.path.join(basepath, "test_data.csv")
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)

    data = pd.concat([data1, data2])
    data.drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last", inplace=True
    )

    return data


def separate_data(data):
    train_data = data[data.answerCode >= 0]
    test_data = data[data.answerCode < 0]

    return train_data, test_data

# train을 train과 valid로 나눔. 단점 : 시간이 1분정도 걸림.
def separate_valid(train_data):
    # 유저 기준 가장 마지막에 본 문제 valid 취급.
    user_final_time = train_data.groupby(['userID']).last()['assessmentItemID']
    train_data['train_valid'] = train_data.apply(lambda x : -1 if x.assessmentItemID == user_final_time[x.userID] else x['answerCode'], axis = 1)
    valid_data = train_data[train_data['train_valid'] == -1]
    train_data = train_data[train_data['train_valid'] >= 0] 
    train_data.drop(['train_valid'], axis = 1, inplace = True)
    valid_data.drop(['train_valid'], axis = 1, inplace = True)
    return train_data, valid_data


def indexing_data(data):
    userid, itemid = (
        sorted(list(set(data.userID))),
        sorted(list(set(data.assessmentItemID))),
    )
    n_user, n_item = len(userid), len(itemid)

    userid_2_index = {v: i for i, v in enumerate(userid)}
    itemid_2_index = {v: i + n_user for i, v in enumerate(itemid)}
    id_2_index = dict(userid_2_index, **itemid_2_index)


    return id_2_index


# 베이스라인 대비 train 인자 추가. train 데이터 or (valid, test) 구분자.
def process_data(data, id_2_index, device, train = True):
    edge, label = [], []
    for user, item, acode in zip(data.userID, data.assessmentItemID, data.answerCode):
        uid, iid = id_2_index[user], id_2_index[item]
        edge.append([uid, iid])
        label.append(acode)

    edge = torch.LongTensor(edge).T  # torch.Size([2, 2468520])
    label = torch.LongTensor(label)  # torch.Size([2468520])
    if train:
        '''
        {'edge': tensor([[    0,     0,     0,  ...,  7439,  7439,  7439],
            [12796, 12797, 12798,  ..., 11170, 11171, 11172]], device='cuda:0'), 'label': tensor([1, 1, 1,  ..., 0, 1, 1], device='cuda:0')}
        '''
        return dict(edge=edge.to(device), label=label.to(device))
    else:
        return dict(edge=edge.to(device), label=label)


def print_data_stat(data, name, logger):
    userid, itemid = list(set(data.userID)), list(set(data.assessmentItemID))
    n_user, n_item = len(userid), len(itemid)

    logger.info(f"{name} Dataset Info")
    logger.info(f" * Num. Users    : {n_user}")
    logger.info(f" * Max. UserID   : {max(userid)}")
    logger.info(f" * Num. Items    : {n_item}")
    logger.info(f" * Num. Records  : {len(data)}")
