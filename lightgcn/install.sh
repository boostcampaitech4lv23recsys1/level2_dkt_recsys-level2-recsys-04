#!/bin/bash
pip install pandas
pip install scikit-learn
pip install wandb
conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg