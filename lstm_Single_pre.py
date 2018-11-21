#  _*_ coding: utf-8 _*_
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def create_dataset(dataset,lock_back=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-lock_back-1):
        a= dataset[i:(i+lock_back),0]
        dataX.append(a)
        dataY.append(dataset[i+lock_back,0])
    return np.array(dataX),np.array(dataY)

def data_pre(min):
    # load_dataset
    # 时间序列做为标签
    if (min == 1):
        dataframe = pd.read_csv(r'E:\data_time_mul\data_wdz_notime.csv', usecols=[3], engine='python',
                                index_col=0)
        # dateframe = dataframe[:24 * 60 * 28]
        train_size = 24 * 60 * 21  # 1分钟
        test_size = 24 * 60 * 28
    elif (min == 5):
        dataframe = pd.read_csv(r'E:\data_time_mul\data_wdz_notime_5min.csv', usecols=[3], engine='python',
                                index_col=0)
        # dateframe = dataframe[:24 * 12 * 28]
        train_size = 24 * 12 * 21  # 5分钟
        test_size = 24 * 60 * 28
    elif (min == 10):
        pd.read_csv(r'E:\data_time_mul\data_wdz_notime_10min.csv', usecols=[3], engine='python', index_col=0)
        # dateframe = dataframe[:24 * 6 * 28]
        train_size = 24 * 6 * 21  # 10分钟
        test_size = 24 * 60 * 28

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataframe)
    train, test = dataset[0:train_size, :], dataset[train_size:test_size, :]
    lock_back = 1
    trainX, trainY = create_dataset(train, lock_back)
    testX, testY = create_dataset(test, lock_back)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    return trainX,testX,trainY,testY,scaler