import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def data_pre(min):
    # load_dataset
    # 时间序列做为标签
    if(min==1):
        dataframe = pd.read_csv(r'E:\data_time_mul\data_wdz_notime.csv',usecols=[0,1,2,3],engine='python',index_col=0)
        # dateframe = dataframe[:24 * 60 * 28]
        train_size = 24 * 60 * 21  # 1分钟
        test_size = 24 * 60 * 28
    elif (min ==5):
        dataframe = pd.read_csv(r'E:\data_time_mul\data_wdz_notime_5min.csv',usecols=[0,1,2,3],engine='python',index_col=0)
        # dateframe = dataframe[:24 * 12 * 28]
        train_size = 24 * 12 * 21 # 5分钟
        test_size = 24 * 60 * 28
    elif (min ==10):
        dataframe = pd.read_csv(r'E:\data_time_mul\data_wdz_notime_10min.csv', usecols=[0, 1, 2, 3], engine='python', index_col=0)
        # dateframe = dataframe[:24 * 6 * 28]
        train_size = 24 * 6 * 21 # 10分钟
        test_size = 24 * 6 * 28

    dataframe_shift = dataframe.shift(-1)
    label = dataframe_shift['YQL']
    dataframe.drop(dataframe.index[len(dataframe) - 1], axis=0, inplace=True)
    label.drop(label.index[len(label) - 1], axis=0, inplace=True)
    x, y = dataframe.values, label.values
    x_scale = MinMaxScaler()
    y_scale = MinMaxScaler()
    X = x_scale.fit_transform(x)
    Y = y_scale.fit_transform(y.reshape(-1, 1))

    X_train, X_test,y_train, y_test = X[:train_size], X[train_size:test_size], Y[:train_size], Y[train_size:test_size]
    X_train = X_train.reshape((-1, 1, 3))
    X_test = X_test.reshape((-1, 1, 3))

    return X_train,X_test,y_train,y_test,y_scale