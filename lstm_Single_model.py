#  _*_ coding: utf-8 _*_
import math
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from lstm_Single_pre import data_pre

trainX,testX,trainY,testY,scaler=data_pre(1)
model=Sequential()
model.add(LSTM(128,return_sequences=True,input_shape=(1,1)))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='mse',optimizer='adam')
model.fit(trainX,trainY,batch_size=72,epochs=500,validation_split=0.1,verbose=1)
print('存入模型中')
model.save('model_1_singletime.h5')
# model.save('model_5_singletime.h5')
# model.save('model_10_singletime.h5')

score =model.evaluate(trainX,trainY)
print('Score:{}'.format(score))

yhat =model.predict(trainX)
yhat = scaler.inverse_transform(yhat)
y_test = scaler.inverse_transform(trainY)
rmse = math.sqrt(mean_squared_error(y_test,yhat))

print('Train RMSE:%.3f' % rmse)