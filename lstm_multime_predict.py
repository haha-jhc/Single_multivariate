# _*_ coding: utf-8 _*_
"""
LSTM prediction
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from lstm_multime_pre import data_pre

str=1#设置时间尺度

if(str==1):
    X_train,X_test,y_train,y_test,y_scale=data_pre(1)
    model = load_model('model_1_multime.h5')
elif(str==5):
    X_train,X_test,y_train,y_test,y_scale=data_pre(5)
    model = load_model('model_5_multime.h5')
if(str==10):
    X_train,X_test,y_train,y_test,y_scale=data_pre(10)
    model = load_model('model_10_multime.h5')
# model = load_model('model_1_multime.h5')
# model = load_model('model_5_multime.h5')
# model = load_model('model_10_multime.h5')

yhat =model.predict(X_test)
yhat = y_scale.inverse_transform(yhat)
y_test = y_scale.inverse_transform(y_test)

rmse = math.sqrt(mean_squared_error(y_test,yhat))
print('Test RMSE:%.3f' % rmse)

plt.rcParams['figure.figsize'] = (20,6)
plt.rcParams['font.sans-serif']=['SimHei']

# print("test")
mape = np.mean(np.abs((y_test-yhat)/y_test))*100
print('Test MAPE:%.3f' % mape)
if(str==1):
    for i in range(0,7):
        label = ["dataset", "testPredict"]
        y_test_plt=y_test[ 24 * 60 * i: 24 * 60 * (i+1)]
        y_hat_plt=yhat[ 24 * 60 * i: 24 * 60 * (i+1)]
        rmse = math.sqrt(mean_squared_error(y_test_plt[:-1], y_hat_plt[1:]))
        print('第%d天'%(i+1),'Test RMSE:%.3f' % rmse)
        mape = np.mean(np.abs((y_test_plt[:-1] - y_hat_plt[1:]) / y_test_plt[:-1])) * 100
        print('第%d天'%(i+1),'Test MAPE:%.3f' % mape)
        l1 = plt.plot(y_test_plt[:-1],color='green')
        l2 = plt.plot(y_hat_plt[1:],color='orange')
        plt.title("预测第%d天的预测结果"%(i+1))
        # plt.show()
        plt.savefig("C:/Users/天津科技大学/Desktop/结果/多变量1分钟/figure_%d.jpg" % (i + 1))
        plt.show()
elif(str==5):
    for i in range(0,7):
        label = ["dataset", "testPredict"]
        y_test_plt=y_test[ 24 * 12 * i: 24 * 12 * (i+1)]
        y_hat_plt=yhat[ 24 * 12 * i: 24 * 12 * (i+1)]
        rmse = math.sqrt(mean_squared_error(y_test_plt[:-1], y_hat_plt[1:]))
        print('第%d天'%(i+1),'Test RMSE:%.3f' % rmse)
        mape = np.mean(np.abs((y_test_plt[:-1] - y_hat_plt[1:]) / y_test_plt[:-1])) * 100
        print('第%d天'%(i+1),'Test MAPE:%.3f' % mape)
        l1 = plt.plot(y_test_plt[:-1],color='green')
        l2 = plt.plot(y_hat_plt[1:],color='orange')
        plt.title("预测第%d天的预测结果"%(i+1))
        # plt.show()
        plt.savefig("C:/Users/天津科技大学/Desktop/结果/多变量5分钟/figure_%d.jpg" % (i + 1))
        plt.show()
elif(str==10):
    for i in range(0,7):
        label = ["dataset", "testPredict"]
        y_test_plt=y_test[ 24 * 6 * i: 24 * 6 * (i+1)]
        y_hat_plt=yhat[ 24 * 6 * i: 24 * 6 * (i+1)]
        rmse = math.sqrt(mean_squared_error(y_test_plt[:-1], y_hat_plt[1:]))
        print('第%d天'%(i+1),'Test RMSE:%.3f' % rmse)
        mape = np.mean(np.abs((y_test_plt[:-1] - y_hat_plt[1:]) / y_test_plt[:-1])) * 100
        print('第%d天'%(i+1),'Test MAPE:%.3f' % mape)
        l1 = plt.plot(y_test_plt[:-1],color='green')
        l2 = plt.plot(y_hat_plt[1:],color='orange')
        plt.title("预测第%d天的预测结果"%(i+1))
        # plt.show()
        plt.savefig("C:/Users/天津科技大学/Desktop/结果/多变量10分钟/figure_%d.jpg" % (i + 1))
        plt.show()