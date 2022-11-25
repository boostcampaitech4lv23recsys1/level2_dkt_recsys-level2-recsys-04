import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import os

class Xg_boost():
    def __init__(self, args):
        self.args = args
        self.model = xgb.XGBClassifier(learning_rate= self.args.learning_rate, n_estimators=self.args.iterations, max_depth = self.args.max_depth)


    def training(self,FEATURE):

        print("###start MODEL training ###")
        self.model.fit(self.train[FEATURE], self.train_value, verbose=100,early_stopping_rounds=100, eval_metric='auc',eval_set=[(self.valid[FEATURE], self.valid_value)])
    
    def make_train_valid_feature(self,data):
        self.train = data[data['answerCode'] >= 0]
        self.test = data[data['answerCode'] < 0]
        user_final_time = self.train.groupby('userID')['Timestamp'].max()
        self.train['train_valid'] = self.train.apply(lambda x : -1 if x.Timestamp == user_final_time[x.userID] else x['answerCode'], axis = 1)
        self.valid = self.train[self.train['train_valid'] < 0]
        self.train = self.train[self.train['train_valid'] >= 0]

        self.train_value = self.train['answerCode']
        self.train.drop(['train_valid'], axis = 1, inplace = True)

        self.valid_value = self.valid['answerCode']
        self.valid.drop(['train_valid'], axis = 1, inplace = True)

    def feature_engineering(self,data, FEATURE):
        # CatBoost에 적용하기 위해선 문자열 데이터로 변환 필요.
        # 카테고리형 feature
        
        le = preprocessing.LabelEncoder()

        for feature in FEATURE:
            if data[feature].dtypes != "float":  # float, str type -> int로 전환
                data[feature] = le.fit_transform(data[feature])
            data[feature] = data[feature].astype("float")

        return data


    def preprocess(self,data,FEATURE):
        print("###start data load & preprocessing###")

        print("FEATURE ENGINEERING")
        data = self.feature_engineering(data,FEATURE)

        print("TRAIN VALID SPLIT")
        self.make_train_valid_feature(data)

    
    

    def inference(self,FEATURE):
        # submission 제출하기 위한 코드
        print("### Inference && Save###")

        _test_pred = self.model.predict_proba(self.test[FEATURE])[:,1]
        self.test['prediction'] = _test_pred
        submission = self.test['prediction'].reset_index(drop = True).reset_index()
        submission.rename(columns = {'index':'id'}, inplace = True)
        submission.to_csv(os.path.join(self.args.output_path, 'submission.csv'), index = False)

