import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,classification_report
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Concatenate, Dropout
from keras.models import Model,Sequential


def dl_train(model, feature_train, target_train, feature_test, target_test,classweight=True,cv=5,  sampling=False,epochs=100, early_stop=10, batch_size=1024):
  oral_col = [i for i in feature_train.columns if 'oral' in i]
  basic_col = [i for i in feature_train.columns if 'oral' not in i]
  early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stop, restore_best_weights=True)
  cvscore_auc = []
  pre_1 = []
  pre_0 = []
  re_1 = []
  re_0 = []
  f1_1 = []
  
  if classweight == True:
    weight_0 = (1/target_train.value_counts()[0]) * (target_train.shape[0]/2.0)
    weight_1 = (1/target_train.value_counts()[1]) * (target_train.shape[0]/2.0)
    class_weight = {0:weight_0, 1:weight_1}
  else:
    class_weight=None
  kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=123)
  for train_index, test_index in kfold.split(feature_train, target_train):
    X_train, X_val = feature_train.iloc[train_index], feature_train.iloc[test_index]
    y_train, y_val = target_train.iloc[train_index], target_train.iloc[test_index]
    X_train_oral, X_val_oral = X_train[oral_col], X_val[oral_col]
    X_train_basic, X_val_basic = X_train[basic_col], X_val[basic_col]
    model.fit([X_train_oral, X_train_basic], y_train, epochs=epochs, validation_data=([X_val_oral,X_val_basic], y_val), batch_size=batch_size, class_weight=class_weight, callbacks=[early_stopping])
    
    X_test_oral, X_test_basic = feature_test[oral_col], feature_test[basic_col]
    predict = model.predict([X_test_oral,X_test_basic])
    label = np.apply_along_axis(lambda x : 1 if x >= 0.5 else 0, 1, predict)
    report = classification_report(target_test, label, output_dict=True)
    score = roc_auc_score(target_test, predict)
    cvscore_auc.append(round(score,4))
    pre_1.append(round(report['1']['precision'],4))
    pre_0.append(round(report['0']['precision'],4))
    re_1.append(round(report['1']['recall'],4))
    re_0.append(round(report['0']['recall'],4))
    f1_1.append(round(report['1']['f1-score'],4))

  return model, round(np.mean(cvscore_auc),4),round(np.mean(pre_1),4),round(np.mean(pre_0),4),round(np.mean(re_1),4),round(np.mean(re_0),4),round(np.mean(f1_1),4),round(np.std(cvscore_auc),4)
