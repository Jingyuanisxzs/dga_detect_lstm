#!/usr/bin/env python
# encoding: utf-8
'''
@author: caopeng
@license: (C) Copyright 2016-2020, Big Bird Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: dga_train.py
@time: 2019/7/10 21:39
@desc:
'''

from model.LstmModel import LstmModel
from model.LstmWithAttentionModel import LstmWithAttentionModel
import codecs
import numpy as np
import datetime

if __name__ == '__main__':
    starttime = datetime.datetime.now()
    batch_size = 50 # 批处理大小
    epochs = 30       # 训练轮数
    trainDataPath = './train_data/dga-feed_train.txt'  # 原始数据文件路径
    modelPath = 'dga_by_lstm_model-attend0107.h5'  # 模型文件保存路径或读取路径

    # 读取配置文件
    charList = {}
    confFilePath = './conf/charList.txt'
    confFile = codecs.open(filename=confFilePath, mode='r', encoding='utf-8', errors='ignore')
    lines = confFile.readlines()
    # 字符序列要从1开始,0是填充字符
    ii = 1
    for line in lines:
        temp = line.strip('\n').strip('\r').strip(' ')
        if temp != '':
            charList[temp] = ii
            ii += 1

    max_features = ii
    #  训练数据
    # 转换数据格式
    x_data_sum = []
    y_data_sum = []
    #
    trainFile = codecs.open(filename=trainDataPath, mode='r', encoding='utf-8', errors='ignore')
    lines = trainFile.readlines()

    groupList = {}
    for line in lines:
        if line[0] == '#':
            continue
        if line.strip('\n').strip('\r').strip(' ') == '':
            continue
        x_data = []
        y_data = [0]*54
        x = line.strip('\n').strip('\r').strip(' ').split(',')[0]
        y = line.strip('\n').strip('\r').strip(' ').split('Domain used by ')[1].split(' for ')[0].split(',')[0].split('(')[0].split(' - ')[0].split(' DGA')[0].lower().strip(' ')
        for char in x:
            try:
                x_data.append(charList[char])
            except:
                print('unexpected char' + ' : ' + char)
                x_data.append(0)

        if not groupList.get(y):
            groupList[y] = len(groupList) + 1

        y_data[groupList[y]-1] = 1

        x_data_sum.append(x_data)
        y_data_sum.append(y_data)

    x_data_sum = np.array(x_data_sum)
    y_data_sum = np.array(y_data_sum)


    # LstmModel
    # lstmModel = LstmModel()
    # lstmModel.trainModel(max_features, x_data_sum, y_data_sum, batch_size, epochs, modelPath)
    lstmWithAttentionModel  = LstmWithAttentionModel()
    lstmWithAttentionModel.trainModel(max_features, x_data_sum, y_data_sum, batch_size, epochs, modelPath)
    endtime = datetime.datetime.now()
    print('=== starttime : ',starttime)
    print('=== endtime   : ',endtime)