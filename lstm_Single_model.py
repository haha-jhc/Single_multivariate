#  _*_ coding: utf-8 _*_
import math
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from lstm_Single_pre import data_pre

trainX,testX,trainY,testY,scaler,train=data_pre(1)
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

trainPredict =model.predict(trainX)
testPredict =model.predict(testX)
#数据反归一化
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore=math.sqrt(mean_squared_error(trainY[0],trainPredict[:,0]))
print('Train Score:%.6f RMSE'%(trainScore))
testScore=math.sqrt(mean_squared_error(testY[0],testPredict[:,0]))
print('Test Score:%.6f RMSE'%(testScore))

