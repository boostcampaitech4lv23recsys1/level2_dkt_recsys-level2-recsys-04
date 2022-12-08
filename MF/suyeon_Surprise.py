#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np


# In[2]:


path = '/opt/ml/input/data'

train = pd.read_csv(path + '/train_data.csv')
test = pd.read_csv(path + '/test_data.csv')


# In[3]:


# train, test 병합. 모든 데이터 활용하기.
dat = pd.concat([train, test], axis = 0)
dat = dat.sort_values(by = ['userID', 'Timestamp'])


# In[4]:


data = dat[['userID', 'assessmentItemID', 'answerCode']].copy().reset_index(drop=True)
data.columns = ['userID', 'itemID', 'rating']


# In[5]:


# rating이 -1로 되어있는 문제를 맞추기 위해 사용.
_train = data[data['rating'] >= 0]
_test = data[data['rating'] < 0]


# In[6]:


_train = _train.drop_duplicates(['userID', 'itemID'], keep='last')


# In[7]:


from surprise import Dataset
from surprise import Reader

reader = Reader(rating_scale=(0,1))
_train_tmp = Dataset.load_from_df(_train.copy(), reader)


# In[8]:


_trainset = _train_tmp.build_full_trainset()


# In[9]:


_trainset


# In[10]:


_train_tmp = train[['userID', 'assessmentItemID', 'answerCode', 'Timestamp']].copy().reset_index(drop=True)
_train_tmp.columns = ['userID', 'itemID', 'rating', 'Timestamp']


# In[12]:


# 유저 마다 가장 마지막 문제를 맞추는 것을 기준으로 평가 하려함
# test 유저에 경우 가장 마지막 문제 전 문제를 맞추는 것을 기준으로 평가함.
user_final_time = _train_tmp.groupby('userID')['Timestamp'].max()
_train_tmp['train_valid'] = _train_tmp.apply(lambda x : -1 if x.Timestamp == user_final_time[x.userID] else x['rating'], axis = 1)
_valid = _train_tmp[_train_tmp['train_valid'] < 0]
_train = _train_tmp[_train_tmp['train_valid'] >= 0]


# In[13]:


_train = _train.drop(columns='Timestamp')
_train = _train.drop(columns='train_valid')


# In[14]:


_train


# In[20]:


_train.nunique()


# In[15]:


_valid = _valid.drop(columns='Timestamp')
_valid = _valid.drop(columns='train_valid')


# In[16]:


_valid


# In[19]:


_valid.nunique()


# In[28]:


_train = Dataset.load_from_df(_train, reader)
trainset = _train.build_full_trainset()


# In[18]:


testset = [_test.iloc[i].to_list() for i in range(len(_test))]


# In[22]:


validset = [_valid.iloc[i].to_list() for i in range(len(_valid))]


# In[23]:


from surprise import SVDpp

model = SVDpp(random_state=42)


# In[24]:


from sklearn.metrics import accuracy_score, roc_auc_score


# In[30]:


n_epochs = [30, 40, 50]
learning_rate = [0.001, 0.005, 0.01]
n_factors = [100, 150, 200]


# In[34]:


cnt = 0

for lr in learning_rate:
    cnt += 1
    print('processing ', cnt, '...')

    model = SVDpp(random_state=42, n_epochs=50, n_factors=150, lr_all=lr)
    
    model.fit(trainset)
    valid_pred = model.test(validset)

    _valid_pred = valid_pred.est

    print(roc_auc_score(_valid.rating, _valid_pred)) # auc
    print(accuracy_score(_valid.rating, np.where(_valid_pred >= 0.5, 1, 0))) # acc, 정확도
    print('process ', cnt, 'learnig_rate = ', lr)


# In[ ]:


model = SVDpp(random_state=42)


# In[ ]:


model.test(testset)


# In[ ]:


pred = model.test(testset)
print('prediction type: ', type(pred))
print('size: ', len(pred))


# In[ ]:


output_list = []
for i in range(len(pred)):
    output_list.append(pred[i].est)


# In[ ]:


submission = pd.DataFrame(columns=['id','prediction'])


# In[ ]:


submission['id'] = list(i for i in range(744))
submission['prediction'] = output_list


# In[ ]:


submission.to_csv('./output/surprise_svd_hp_tunned.csv', index = False)

