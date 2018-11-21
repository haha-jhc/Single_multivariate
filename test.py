import math
from lstm_multime_pre import data_pre
from keras.models import load_model
from sklearn.metrics import mean_squared_error

X_train,X_test,y_train,y_test,y_scale=data_pre(1)

model = load_model('model_1_multime.h5')

y_hat1 =model.predict(X_train)
y_hat1 = y_scale.inverse_transform(y_hat1)
y_train = y_scale.inverse_transform(y_train)
train_rmse = math.sqrt(mean_squared_error(y_train,y_hat1))
print('Train Score:%.6f RMSE'%(train_rmse))

y_hat2 =model.predict(X_test)
y_hat2 = y_scale.inverse_transform(y_hat2)
y_test = y_scale.inverse_transform(y_test)
test_rmse = math.sqrt(mean_squared_error(y_test,y_hat2))
print('Test Score:%.6f RMSE'%(test_rmse))