import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.nn.models import LightGCN
from torch.optim import lr_scheduler

def build(n_node, weight=None, logger=None, **kwargs):
    """
    Args:
        n_node (int): id+item 개수.
        weight (_type_, optional): ??
        logger (_type_, optional): ??
        **kwargs : embedding_dim, num_layers, alpha, 기타 추가 가능(LightGCN 파라미터)
    Returns:
        model : 바로 사용 가능한 모델 출력
    """    
    model = LightGCN(n_node, **kwargs)
    if weight:
        if not os.path.isfile(weight):
            logger.fatal("Model Weight File Not Exist")
        logger.info("Load model")
        state = torch.load(weight)["model"]
        model.load_state_dict(state)
        return model
    # weight가 정확히 무슨 역할을 하는진 모르겠으나 else로 계속 실행됨.
    else:
        logger.info("No load model")
        return model


def train(
    model,
    train_data, 
    valid_data=None,
    n_epoch=100,
    learning_rate=0.01,
    lr_decay = 10,
    gamma = 0.9,
    use_wandb=False,
    weight=None,
    logger=None,
):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=gamma)
    if not os.path.exists(weight):
        os.makedirs(weight)

    if valid_data is None:
        eids = np.arange(len(train_data["label"]))
        # train_data["label"] 중 1000개 뽑기.
        eids = np.random.permutation(eids)[:1000]
        edge, label = train_data["edge"], train_data["label"]
        label = label.to("cpu").detach().numpy()
        # train 내에서 valid 따로 추출. 즉 train 데이터 일부가 valid.
        # 개인적으로 valid가 train과 구분되지 않아 이상해질 것 같음.
        # valid는 test와 같이 학습과정에서 사용되지 않아야하는데 그렇지 못함.
        valid_data = dict(edge=edge[:, eids], label=label[eids])

    logger.info(f"Training Started : n_epoch={n_epoch}")
    best_auc, best_epoch = 0, -1
    for e in range(n_epoch):
        # forward
        pred = model(train_data["edge"])

        # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/lightgcn.html
        loss = model.link_pred_loss(pred, train_data["label"])

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            prob = model.predict_link(valid_data["edge"], prob=True)
            prob = prob.detach().cpu().numpy()
            acc = accuracy_score(valid_data["label"], prob > 0.5)
            auc = roc_auc_score(valid_data["label"], prob)
            logger.info(
                f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}"
            )
            if use_wandb:
                import wandb
                wandb.log(dict(loss=loss, acc=acc, auc=auc))

        if weight:
            if auc > best_auc:
                logger.info(
                    f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}, Best AUC"
                )
                best_auc, best_epoch = auc, e
                torch.save(
                    {"model": model.state_dict(), "epoch": e + 1},
                    os.path.join(weight, f"best_model.pt"),
                )

    scheduler.step()
    torch.save(
        {"model": model.state_dict(), "epoch": e + 1},
        os.path.join(weight, f"last_model.pt"),
    )
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")


def inference(model, data, logger=None):
    model.eval()
    with torch.no_grad():
        pred = model.predict_link(data["edge"], prob=True)
        return pred
