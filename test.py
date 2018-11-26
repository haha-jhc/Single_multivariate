import math
import numpy as np
from lstm_Single_pre import data_pre
from keras.models import load_model
from sklearn.metrics import mean_squared_error

# X_train,X_test,y_train,y_test,y_scale=data_pre(1)

model = load_model('model_5_singletime.h5')

# y_hat1 =model.predict(X_train)
# y_hat1 = y_scale.inverse_transform(y_hat1)
# y_train = y_scale.inverse_transform(y_train)
# train_rmse = math.sqrt(mean_squared_error(y_train,y_hat1))
# print('Train Score:%.6f RMSE'%(train_rmse))
#
# y_hat2 =model.predict(X_test)
# y_hat2 = y_scale.inverse_transform(y_hat2)
# y_test = y_scale.inverse_transform(y_test)
# test_rmse = math.sqrt(mean_squared_error(y_test,y_hat2))
# print('Test Score:%.6f RMSE'%(test_rmse))
trainX,testX,trainY,testY,scaler,train,test,train_size=data_pre(1)
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
# testScore=math.sqrt(mean_squared_error(testY[0][:-1],testPredict[1:,0]))
# print('Test Score:%.6f RMSE'%(testScore))
# mape2 = np.mean(np.abs((testY[0][:-1]-testPredict[1:,0])/testY[0][:-1]))*100
# print('Test MAPE:%.3f' % mape2)