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

    print("loading csv.....")
    ### HSEUNEH_SAINT.ipynb 돌리면 "전처리" + "증강" csv 생성 ###

    train_df = pd.read_csv(Config.TRAIN_FILE)
    test_df = pd.read_csv(Config.TEST_FILE)

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
    
    # breakpoint()


    print("splitting")
    # Test 데이터의 user가 Valid로 들어가면, Test user가 학습되지 않기 때문에, 최초의 Train 데이터에서 Split 진행
    train, val = train_test_split(train_group, test_size=Config.VALID_SIZE, shuffle=True)
    train = pd.concat([train, test_to_train_group])  # Test에서 -1 제외한 user 별 데이터 Train으로 concat
    test = test_group.copy()

    # breakpoint()

    # 메모리 청소하는 부분인듯? GKT 모델에도 써보면 좋을듯 -> 이미 있음
    del train_df, test_df, test_to_train_df, train_group, test_group, test_to_train_group
    gc.collect()
 


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
