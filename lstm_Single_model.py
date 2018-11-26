#  _*_ coding: utf-8 _*_
import math
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from lstm_Single_pre import data_pre

str=1#设置时间刻度

if(str==1):
    trainX,testX,trainY,testY,scaler,train,test,train_size=data_pre(1)
elif(str==5):
    trainX, testX, trainY, testY, scaler, train, test, train_size = data_pre(5)
elif(str==10):
    trainX, testX, trainY, testY, scaler, train, test, train_size = data_pre(10)
model=Sequential()
model.add(LSTM(128,return_sequences=True,input_shape=(1,1)))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

start = time.time()
model.compile(loss='mse',optimizer='adam')
model.fit(trainX,trainY,batch_size=72,epochs=500,validation_split=0.1,verbose=1)
print("> Compilation Time :",time.time()-start)
print('存入模型中')
if(str==1):
    model.save('model_1_singletime.h5')
elif(str==5):
    model.save('model_5_singletime.h5')
elif(str==10):
    model.save('model_10_singletime.h5')

trainPredict =model.predict(trainX)
testPredict =model.predict(testX)
#数据反归一化
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore=math.sqrt(mean_squared_error(trainY[0],trainPredict[:,0]))
print('Train Score:%.6f RMSE'%(trainScore))
mape1 = np.mean(np.abs((trainY[0]-trainPredict[:,0])/trainY[0]))*100
print('Train MAPE:%.3f' % mape1)
testScore=math.sqrt(mean_squared_error(testY[0],testPredict[:,0]))
print('Test Score:%.6f RMSE'%(testScore))
mape2 = np.mean(np.abs((testY[0]-testPredict[:,0])/testY[0]))*100
print('Test MAPE:%.3f' % mape2)


