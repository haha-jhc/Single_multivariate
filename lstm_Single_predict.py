# _*_ coding: utf-8 _*_
"""
LSTM prediction
"""
import math
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.pyplot import savefig
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from lstm_Single_pre import data_pre

str=1 #设置时间刻度

if(str==1):
    trainX,testX,trainY,testY,scaler,train,test,train_size=data_pre(1)
    model = load_model('model_1_singletime.h5')
if(str==5):
    trainX,testX,trainY,testY,scaler,train,test,train_size=data_pre(5)
    model = load_model('model_5_singletime.h5')
if(str==10):
    trainX,testX,trainY,testY,scaler,train,test,train_size=data_pre(10)
    model = load_model('model_10_singletime.h5')
# model = load_model('model_1_singletime.h5')
# model = load_model('model_5_singletime.h5')
# model = load_model('model_10_singletime.h5')

plt.rcParams['figure.figsize'] = (20,6)
plt.rcParams['font.sans-serif']=['SimHei']

# print("test")
# label = ["dataset", "testPredict"]
# y_test_plt=y_test[ : 24 * 60 ]
# y_hat_plt=yhat[ : 24 * 60]
# l1 = plt.plot(y_test_plt,color='green')
# l2 = plt.plot(y_hat_plt[1:],color='orange')
# plt.title("预测第1天的预测结果")
# plt.show()
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
#数据反归一化
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore=math.sqrt(mean_squared_error(trainY[0],trainPredict[:,0]))
print('Train Score:%.6f RMSE'%(trainScore))
testScore=math.sqrt(mean_squared_error(testY[0],testPredict[:,0]))
print('Test Score:%.6f RMSE'%(testScore))
mape = np.mean(np.abs((testY[0]-testPredict[:,0])/testY[0]))*100
print('Test MAPE:%.3f' % mape)

train = train[:train_size-1]
trainPredictPlot =np.empty_like(train)
trainPredictPlot[:,:] =np.nan
trainPredictPlot[1:len(trainPredictPlot)+1,:]=trainPredict

#shift test predictions for plotting
testPredictPlot = np.empty_like(test)
testPredictPlot[:,:] = np.nan
testPredictPlot = testPredict
print(len(testPredictPlot))
print(len(test))
y_test=scaler.inverse_transform(test)
if(str==1):
    for i in range(0,7):
        label = ["dataset", "testPredict"]
        if(i<6):
            y_test_plt=y_test[ 24 * 60 * i: 24 * 60 * (i+1)]
            y_hat_plt=testPredictPlot[ 24 * 60 * i: 24 * 60 * (i+1)]
        else:
            y_test_plt = y_test[24 * 60 * i:-2]
            y_hat_plt = testPredictPlot[24 * 60 * i: ]
        # rmse = math.sqrt(mean_squared_error(y_test_plt[:-1], y_hat_plt[1:]))
        # print('第%d天' % (i + 1), 'Test RMSE:%.3f' % rmse)
        # mape = np.mean(np.abs((y_test_plt[:-1] - y_hat_plt[1:]) / y_test_plt[:-1])) * 100
        # print('第%d天' % (i + 1), 'Test MAPE:%.3f' % mape)
        rmse = math.sqrt(mean_squared_error(y_test_plt, y_hat_plt))
        print('第%d天' % (i + 1), 'Test RMSE:%.3f' % rmse)
        mape = np.mean(np.abs((y_test_plt - y_hat_plt) / y_test_plt)) * 100
        print('第%d天' % (i + 1), 'Test MAPE:%.3f' % mape)
        l1 = plt.plot(y_test_plt,color='green')
        l2 = plt.plot(y_hat_plt,color='orange')
        plt.title("预测第%d天的预测结果"%(i+1))
        # plt.show()
        plt.savefig("C:/Users/天津科技大学/Desktop/结果/单变量1分钟/figure_%d.jpg"%(i+1))
        plt.show()
if(str==5):
    for i in range(0,7):
        label = ["dataset", "testPredict"]
        if(i<6):
            y_test_plt=y_test[ 24 * 12 * i: 24 * 12 * (i+1)]
            y_hat_plt=testPredictPlot[ 24 * 12 * i: 24 * 12 * (i+1)]
        else:
            y_test_plt = y_test[24 * 12 * i:-2]
            y_hat_plt = testPredictPlot[24 * 12 * i: ]
        # rmse = math.sqrt(mean_squared_error(y_test_plt[:-1], y_hat_plt[1:]))
        # print('第%d天' % (i + 1), 'Test RMSE:%.3f' % rmse)
        # mape = np.mean(np.abs((y_test_plt[:-1] - y_hat_plt[1:]) / y_test_plt[:-1])) * 100
        # print('第%d天' % (i + 1), 'Test MAPE:%.3f' % mape)
        rmse = math.sqrt(mean_squared_error(y_test_plt, y_hat_plt))
        print('第%d天' % (i + 1), 'Test RMSE:%.3f' % rmse)
        mape = np.mean(np.abs((y_test_plt - y_hat_plt) / y_test_plt)) * 100
        print('第%d天' % (i + 1), 'Test MAPE:%.3f' % mape)
        l1 = plt.plot(y_test_plt,color='green')
        l2 = plt.plot(y_hat_plt,color='orange')
        plt.title("预测第%d天的预测结果"%(i+1))
        # plt.show()
        plt.savefig("C:/Users/天津科技大学/Desktop/结果/单变量5分钟/figure_%d.jpg"%(i+1))
        plt.show()
if(str==10):
    for i in range(0,7):
        label = ["dataset", "testPredict"]
        if(i<6):
            y_test_plt=y_test[ 24 * 6 * i: 24 * 6 * (i+1)]
            y_hat_plt=testPredictPlot[ 24 * 6 * i: 24 * 6 * (i+1)]
        else:
            y_test_plt = y_test[24 * 6 * i:-2]
            y_hat_plt = testPredictPlot[24 * 6 * i: ]
        # rmse = math.sqrt(mean_squared_error(y_test_plt[:-1], y_hat_plt[1:]))
        # print('第%d天' % (i + 1), 'Test RMSE:%.3f' % rmse)
        # mape = np.mean(np.abs((y_test_plt[:-1] - y_hat_plt[1:]) / y_test_plt[:-1])) * 100
        # print('第%d天' % (i + 1), 'Test MAPE:%.3f' % mape)
        rmse = math.sqrt(mean_squared_error(y_test_plt, y_hat_plt))
        print('第%d天' % (i + 1), 'Test RMSE:%.3f' % rmse)
        mape = np.mean(np.abs((y_test_plt - y_hat_plt) / y_test_plt)) * 100
        print('第%d天' % (i + 1), 'Test MAPE:%.3f' % mape)
        l1 = plt.plot(y_test_plt,color='green')
        l2 = plt.plot(y_hat_plt,color='orange')
        plt.title("预测第%d天的预测结果"%(i+1))
        # plt.show()
        plt.savefig("C:/Users/天津科技大学/Desktop/结果/单变量10分钟/figure_%d.jpg"%(i+1))
        plt.show()