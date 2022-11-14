import os

import torch
from args import parse_args
from src import trainer
from src.dataloader import Preprocess


def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess = Preprocess(args)
    preprocess.load_test_data(args.test_file_name)
    test_data = preprocess.get_test_data()
    model = trainer.get_model(args).to(args.device)
    trainer.inference(args, test_data, model)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
