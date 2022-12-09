from datetime import datetime
from pytz import timezone
# ====================================================
# CFG
# ====================================================
class CFG:
    use_cuda_if_available = True
    user_wandb = True
    # wandb_kwargs = dict(
    #     entity="schini",
    #     project="sweep-test-lightgcn"
    # )

    # data
    basepath = "/opt/ml/input/data/"
    loader_verbose = True

    # dump
    output_dir = "./output/"
    pred_file = "submission_1128.csv"

    # build
    embedding_dim = 512  # int, 64
    num_layers = 3  # int, 1
    alpha = None  # Optional[Union[float, Tensor]]
    build_kwargs = {}  # other arguments
    weight = "./weight/best_model.pt"

    # train
    n_epoch = 150 # 30
    learning_rate = 0.005371  # 0.05
    # lr_decay, gamma는 추가한 것, 스케줄러.
    lr_decay = 5 
    gamma = 0.9
    weight_basepath = "./weight"


sweep_configuration = {
    'method': 'random',
    'entity': 'schini',
    'project': 'sweep-test-lightgcn',
    'name': datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d-%H-%M-%S'),
    'metric': {'goal': 'maximize', 'name': 'auc'},
    'parameters': 
        {
            # 'batch_size': {'values': [16, 32, 64]},
            'n_epoch': {'values': [5, 10]},
            'learning_rate': {'max': 0.1, 'min': 0.0001}
        }
}

logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "run.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}
