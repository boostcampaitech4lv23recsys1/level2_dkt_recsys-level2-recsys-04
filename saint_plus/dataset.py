from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from config import Config
from sklearn.model_selection import train_test_split
import gc


class DKTDataset(Dataset):
    def __init__(self, samples, max_seq):
        super().__init__()
        self.samples = samples
        self.max_seq = max_seq
        self.data = []
        for id in self.samples.index:
            exe_ids, answers, ela_time, categories = self.samples[id]
            input_answers = answers[:] + 1  ## padding과 start-token 0 값과 구별하기 위해, 틀린 문제 - 1 /  맞은 문제 - 2 로 설정
            if len(exe_ids) > max_seq:
                self.data.append((exe_ids[-max_seq:], answers[-max_seq:], input_answers[-max_seq:], ela_time[-max_seq:], categories[-max_seq:]))
                # if is_test:  # Test 데이터의 경우 증강하면 안되기 때문에, 마지막 max_seq 길이만큼의 데이터만 가져옴
                #     self.data.append((exe_ids[-max_seq:], answers[-max_seq:], ela_time[-max_seq:], categories[-max_seq:]))
                # else:
                #     for l in range(len(exe_ids)-max_seq):  # max_seq이 100이면, 0~99 / 1~100 / 2~101 / ... 이런 식으로 잘라서 데이터 증강
                #         self.data.append(
                #             (exe_ids[l:l+max_seq], answers[l:l+max_seq], ela_time[l:l+max_seq], categories[l:l+max_seq]))
            elif len(exe_ids) <= self.max_seq and len(exe_ids) > Config.MIN_SEQ:
                self.data.append((exe_ids, answers, input_answers, ela_time, categories))
            else:
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question_ids, answers, input_answers, ela_time, exe_category = self.data[idx]
        seq_len = len(question_ids)
        exe_ids = np.zeros(self.max_seq, dtype=int)
        ans = np.zeros(self.max_seq, dtype=int)
        input_ans = np.zeros(self.max_seq, dtype=int)
        elapsed_time = np.zeros(self.max_seq, dtype=int) # normalize 했으면 float
        exe_cat = np.zeros(self.max_seq, dtype=int)

        # exe_ids[:seq_len] = question_ids  # MHA에서 Mask 모양 고려했을 때, 이게 맞는 것 같음
        # ans[:seq_len] = answers
        # elapsed_time[:seq_len] = ela_time
        # exe_cat[:seq_len] = exe_category
        if seq_len < self.max_seq:
            exe_ids[-seq_len:] = question_ids
            ans[-seq_len:] = answers[:]
            input_ans[-seq_len:] = input_answers[:]
            elapsed_time[-seq_len:] = ela_time
            exe_cat[-seq_len:] = exe_category
        else:
            exe_ids[:] = question_ids[-self.max_seq:]
            ans[:] = answers[-self.max_seq:]
            input_ans[:] = input_answers[-self.max_seq:]
            elapsed_time[:] = ela_time[-self.max_seq:]
            exe_cat[:] = exe_category[-self.max_seq:]

        # 한 칸 앞으로 당기는거 이미 앞에서 진행한 작업이라 skip 해놓음
        # 이라고 생각했는데 아님, 이거 디코더 인풋 타임스탬프 하나 땡겨주는 작업이었음
        # 정답고 시간 태그 타임스탬프 한 칸 땡겨주기
        input_rtime = np.zeros(self.max_seq, dtype=int)
        input_rtime = np.insert(elapsed_time, 0, -1)       
        input_rtime = np.delete(input_rtime, -1)

        input_ans = np.insert(input_ans, 0, 0)       
        input_ans = np.delete(input_ans, -1)
        self.input = {"input_ids": exe_ids, "input_rtime": input_rtime, "input_cat": exe_cat}
        return self.input, input_ans, ans


def get_dataloaders():
    dtypes = {
        'userID': 'int16',
        'answerCode': 'int8',
        'KnowledgeTag': 'int16',
    }
    print("loading csv.....")
    train_df = pd.read_csv(Config.TRAIN_FILE, usecols=[0, 1, 3, 4, 5], dtype=dtypes, parse_dates=['Timestamp'])  # TestID 빼고 볼러옴
    test_df = pd.read_csv(Config.TEST_FILE, usecols=[0, 1, 3, 4, 5], dtype=dtypes, parse_dates=['Timestamp'])
    train_df = train_df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
    test_df = test_df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
    
    train_df['assessmentItemID'], _ = pd.factorize(train_df['assessmentItemID'], sort=True)
    train_df['KnowledgeTag'], _ = pd.factorize(train_df['KnowledgeTag'], sort=True)
    test_df['assessmentItemID'], _ = pd.factorize(test_df['assessmentItemID'], sort=True)
    test_df['KnowledgeTag'], _ = pd.factorize(test_df['KnowledgeTag'], sort=True)
    train_df['assessmentItemID'] += 1  # padding 한 0 값이랑, 문제 라벨 0 이랑 구분하기 위해서 1 더해줌
    test_df['assessmentItemID'] += 1

    elapse = train_df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff(periods=1)['Timestamp']
    elapse = elapse.fillna(pd.Timedelta(seconds=0)).apply(lambda x: x.total_seconds()).astype(np.int32)
    elapse = elapse.apply(lambda x: x if x <= Config.MAX_EPLAPSED_TIME else Config.MAX_EPLAPSED_TIME)
    elapse /= Config.MAX_EPLAPSED_TIME  # Normalize (ex. 0~600 -> 0~1로 바꿔줌) why? 나중에 임베딩 벡터에 600 곱해지면 너무 커지니까. 근데 이거 안하고 해봐도 좋을듯
    train_df['prior_question_elapsed_time'] = elapse
    elapse = test_df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff(periods=1)['Timestamp']
    elapse = elapse.fillna(pd.Timedelta(seconds=0)).apply(lambda x: x.total_seconds()).astype(np.int32)
    elapse = elapse.apply(lambda x: x if x <= Config.MAX_EPLAPSED_TIME else Config.MAX_EPLAPSED_TIME)
    elapse /= Config.MAX_EPLAPSED_TIME
    test_df['prior_question_elapsed_time'] = elapse

    

    # grouping based on userID to get the data supplu
    print("Grouping users...")
    train_group = train_df[["userID", "assessmentItemID", "answerCode", "prior_question_elapsed_time", "KnowledgeTag"]]\
        .groupby("userID")\
        .apply(lambda r: (r.assessmentItemID.values, r.answerCode.values,
                          r.prior_question_elapsed_time.values, r.KnowledgeTag.values))

    test_to_train_df = test_df.loc[test_df['answerCode'] != -1]  # Test 데이터 user 별 마지막 문제 제외한 데이터
    test_to_train_group = test_to_train_df[["userID", "assessmentItemID", "answerCode", "prior_question_elapsed_time", "KnowledgeTag"]]\
        .groupby("userID")\
        .apply(lambda r: (r.assessmentItemID.values, r.answerCode.values,
                            r.prior_question_elapsed_time.values, r.KnowledgeTag.values))

    # test_df.loc[test_df['answerCode'] == -1, 'answerCode'] = 2  # -1 그대로 두면 embedding 단계에서 error 발생
    test_group = test_df[["userID", "assessmentItemID", "answerCode", "prior_question_elapsed_time", "KnowledgeTag"]]\
        .groupby("userID")\
        .apply(lambda r: (r.assessmentItemID.values, r.answerCode.values,
                          r.prior_question_elapsed_time.values, r.KnowledgeTag.values))
    

    print("splitting")
    # Test 데이터의 user가 Valid로 들어가면, Test user가 학습되지 않기 때문에, 최초의 Train 데이터에서 Split 진행
    train, val = train_test_split(train_group, test_size=Config.VALID_SIZE, shuffle=True)
    train = pd.concat([train, test_to_train_group])  # Test에서 -1 제외한 user 별 데이터 Train으로 concat
    test = test_group.copy()

    #data augmentation
    if Config.DATA_AUG:
        train_origin = train.copy()
        n= 1
        print(f'======origin length      : {len(train)}======')
        breakpoint()
        for i in range(Config.DATA_AUG):
            print(f'START {n}th AUGMENTATION')
            train_origin['userID'] += 7439+1
            train = pd.concat([train, train_origin], axis = 0)
            print(f'END   {n}th AUGMENTATION')
            n += 1
        print(f'======after augmentation : {len(train)}======')
        breakpoint()
    # 메모리 청소하는 부분인듯? GKT 모델에도 써보면 좋을듯 -> 이미 있음
    del train_df, test_df, test_to_train_df, elapse, train_group, test_group, test_to_train_group
    gc.collect()


    # print("shape of train :", train_df.shape)
    # print("shape of val :", test_df.shape)

    # train_df = train_df[train_df.content_type_id == 0]
    # train_df.prior_question_elapsed_time.fillna(0, inplace=True)
    # train_df.prior_question_elapsed_time /= 1000
    # train_df.prior_question_elapsed_time.clip(lower=0,upper=300,inplace=True)
    # train_df.prior_question_elapsed_time = train_df.prior_question_elapsed_time.astype(np.int)
    
    # train_df = train_df.sort_values(["timestamp"], ascending=True).reset_index(drop=True)
    # n_skills = train_df.assessmentItemID.nunique()
    # print("no. of skills :", n_skills)
    # print("shape after exlusion:", train_df.shape)

    train_dataset = DKTDataset(train, max_seq=Config.MAX_SEQ)
    val_dataset = DKTDataset(val, max_seq=Config.MAX_SEQ)
    test_dataset = DKTDataset(test, max_seq=Config.MAX_SEQ)

    train_loader = DataLoader(train_dataset,
                              batch_size=Config.BATCH_SIZE,
                              num_workers=8,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=Config.BATCH_SIZE,
                            num_workers=8,
                            shuffle=False)
    test_loader = DataLoader(test_dataset,
                            batch_size=Config.BATCH_SIZE,  # len(train_dataset)
                            num_workers=8,
                            shuffle=False)

    del train_dataset, val_dataset, test_dataset
    gc.collect()
    return train_loader, val_loader, test_loader
