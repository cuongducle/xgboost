import pickle
import xgboost as xgb
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

data = pickle.load(open('feature/feature_new.pkl', "rb" ))
label = pickle.load(open('feature/label_new.pkl', "rb" ))


y_train = data[-1000:]
y_test = label[-1000:]

encoder = preprocessing.LabelEncoder()
encoder.fit(label)
label_make = encoder.transform(label)
label_test = encoder.transform(y_test)

label_list = list(set(list(label)))

label = np.asarray(label_make)
data = np.asarray(data)
y_train = np.asarray(y_train)
y_test = np.asarray(label_test)

dtrain = xgb.DMatrix(data, label=label)
dtest = xgb.DMatrix(y_train, label=y_test)

print("load done")
param = {'max_depth': 1000}
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 1
bst = xgb.train(param,dtrain,num_round,watchlist)

evals_result = {}
bst = xgb.train(param, dtrain, num_round, watchlist)

bst.save_model('model/xgboost_java.model')