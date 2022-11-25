# GKT
The implementation of the paper [Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network](https://dl.acm.org/doi/10.1145/3350546.3352513).

The architecture of the GKT is as follows:

![](gkt_architecture.png)

## Setup

To run this code you need the following:

- a machine with GPUs
- python3
- numpy, pandas, scipy, scikit-learn and torch packages:
```
pip3 install numpy==1.17.4 pandas==1.1.2 scipy==1.5.2 scikit-learn==0.23.2 torch==1.4.0
```

## Training the model

Use the `train.py` script to train the model. To train the GKT model on ASSISTments2009-2010 skill-builder dataset, simply use:

```
python3 train.py
```
