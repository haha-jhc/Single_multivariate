# _*_ coding: utf-8 _*_
"""
LSTM prediction
"""
import math
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from lstm_multime_pre import data_pre

X_train,X_test,y_train,y_test,y_scale=data_pre(1)

model = load_model('model_1_multime.h5')
# model = load_model('model_5_multime.h5')
# model = load_model('model_10_multime.h5')

yhat =model.predict(X_test)
yhat = y_scale.inverse_transform(yhat)
y_test = y_scale.inverse_transform(y_test)

rmse = math.sqrt(mean_squared_error(y_test,yhat))
print('Test RMSE:%.3f' % rmse)

plt.rcParams['figure.figsize'] = (20,6)
plt.rcParams['font.sans-serif']=['SimHei']

print("出图")
label = ["dataset", "testPredict"]
y_test_plt=y_test[ : 24 * 60 ]
y_hat_plt=yhat[ : 24 * 60]
l1 = plt.plot(y_test_plt,color='green')
l2 = plt.plot(y_hat_plt[1:],color='orange')
plt.title("预测第1天的预测结果")
plt.show()
