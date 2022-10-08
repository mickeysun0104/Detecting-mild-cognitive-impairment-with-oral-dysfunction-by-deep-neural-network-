import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Concatenate, Dropout
from keras.models import Model,Sequential

def residual_layer(X_input,units):
  X1 = keras.layers.Dense(units, activation='relu')(X_input)
  X2 = keras.layers.Dense(X_input.shape[1])(X1)
  X_out = tf.keras.layers.ReLU()(keras.layers.Add()([X_input,X2]))
  return X_out

def deepcrossing(emb_dim, residual_units, feature):
    oral_col = [i for i in feature.columns if 'oral' in i]
    basic_col = [i for i in feature.columns if 'oral' not in i]
    #create input
    oral_in = keras.layers.Input(shape=(len(oral_col),), name='oral_features')
    basic_in = keras.layers.Input(shape=(len(basic_col),), name='basic_features')

    #embedding layer, emb_dim=how many dimentions we want a feature to have
    embedding = keras.layers.Dense(emb_dim, activation='relu',name='Embedding', use_bias=False)(oral_in)  

    #Stacking layer
    stacking = keras.layers.Concatenate(name='stacking')([embedding, basic_in])


    #Residual layer, put the number of neurons in the list to construct the correspondant dense layer
    for i,res_num in enumerate(residual_units):
      if i == 0:
        res_out = residual_layer(stacking, res_num)
      else:
        res_out = residual_layer(res_out, res_num)

    #scoring layer
    scoring = keras.layers.Dense(1, activation='sigmoid', name='scoring')(res_out)

    model = keras.models.Model([oral_in, basic_in], scoring)
    return model