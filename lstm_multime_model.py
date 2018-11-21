# _*_ coding: utf-8 _*_
"""
LSTM prediction
考虑时间维度，以一天为周期输入分钟数
"""
import math
from keras.layers.core import Dense,Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from lstm_multime_pre import data_pre

X_train,X_test,y_train,y_test,y_scale=data_pre(1)

model=Sequential()
model.add(LSTM(128,return_sequences=True,input_shape=(1,3)))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,batch_size=72,epochs=500,validation_split=0.1,verbose=1)
print('存入模型中')
model.save('model_1_multime.h5')
# model.save('model_5_multime.h5')
#model.save('model_10_multime.h5')

score =model.evaluate(X_train,y_train)
print('Score:{}'.format(score))

yhat =model.predict(X_train)
yhat = y_scale.inverse_transform(yhat)
y_test = y_scale.inverse_transform(y_train)
rmse = math.sqrt(mean_squared_error(y_test,yhat))

print('Train RMSE:%.3f' % rmse)