# ====================================================
# CFG
# ====================================================
class CFG:
    use_cuda_if_available = True
    user_wandb = False
    wandb_kwargs = dict(project="dkt-gcn")

    # data
    basepath = "/opt/ml/input/data/"
    loader_verbose = True

    # dump
    output_dir = "./output/"
    pred_file = "submission_1128.csv"

    # build
    embedding_dim = 64  # int, 64
    num_layers = 3  # int, 1
    alpha = None  # Optional[Union[float, Tensor]]
    build_kwargs = {}  # other arguments
    weight = "./weight/best_model.pt"

    # train
    batch_size = 2048
    n_epoch = 200 # 30
    learning_rate = 0.01 # 0.05
    weight_decay = 5e-7
    # lr_decay, gamma는 추가한 것, 스케줄러.
    lr_decay = 10
    gamma = 0.6
    weight_basepath = "./weight" 


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
