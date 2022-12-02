import torch


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    MAX_SEQ = 100  # USER 별 최대 문제 풀이 개수
    MIN_SEQ = 13   # USER 별 최소 문제 풀이 개수 (대회 Test 데이터 최소 문제 풀이 유저 -> 15개)
    EMBED_DIMS = 512
    ENC_HEADS = DEC_HEADS = 1
    NUM_ENCODER = NUM_DECODER = 1
    EPOCHS = 1
    BATCH_SIZE = 32
    TRAIN_FILE = "/opt/ml/input/data/train_data.csv"
    TEST_FILE = "/opt/ml/input/data/test_data.csv"
    SUBMISSION_FILE = "/opt/ml/input/data/sample_submission.csv"
    TOTAL_EXE = 9454 + 1  # 대회 dataset 서로 다른 assessmentItemID 개수, +1은 mask 0 때문에 해줌
    TOTAL_CAT = 912   # 대회 dataset 서로 다른 KnowledgeTag 개수
    MAX_EPLAPSED_TIME = 600  # elapsed_time = min(MAX_ELAPSED_TIME, (현재 문제 풀이시간 - 이전 문제 풀이시간))
    VALID_SIZE = 0.1  # Train과 Valid split 비율
