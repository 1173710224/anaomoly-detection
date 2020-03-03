'''
author:
cbz
'''
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.callbacks import ReduceLROnPlateau
from process import makedata
from util import *
import pandas as pd
import numpy as np
from queue import Queue
import time
import json
'''
train method
1:311
2:bikeinflow
3:bikeoutflow
4:taxiinflow
5:taxiouflow
'''
def train1(region):
    path = 'model/' + str(region) + '311.h5'
    file = open(path,'w')
    rawdata = pd.read_csv('model/' + str(region) + '311.csv').values
    if len(rawdata) == 0:
        print('Region' + str(region) + ' don\'t have data about 311!\nUse the unified model!')
        return
    dic = makedata(region,'model/' + str(region) + '311.csv')
    x = np.array(dic['x'])
    y = np.array(dic['y'])

    model = None
    if os.path.getsize('model/' + str(region) + '311.h5') == 0:
        model = Sequential()
        model.add(Dense(10,activation='sigmoid',use_bias=True))
        model.add(Dense(1,use_bias=True))
        model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.RMSprop(0.1))
    else:
        model = load_model('model/' + str(region) + '311.h5')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
    model.fit(x=x, y=y, epochs=100, validation_split=0.1, verbose = 0, shuffle=True, batch_size=128,
              callbacks=[reduce_lr])
    model.save('model/' + str(region) + '311.h5')
    print(str(region) + '-311:100 epoches have been trained over!')
    return
def train2(region):
    # path = 'model/' + str(region) + '/bikeinflow.h5'
    # file = open(path, 'w')
    rawdata = pd.read_csv('model/' + str(region) + '/bikeinflow.csv').values
    if len(rawdata) == 0:
        print('Region' + str(region) + ' don\'t have data about bikeinflow!\nUse the unified model!')
    #     return
    dic = makedata(region, 'model/' + str(region) + '/bikeinflow.csv')
    x = np.array(dic['x'])
    y = np.array(dic['y'])

    model = None
    if os.path.getsize('model/' + str(region) + '/bikeinflow.h5') == 0:
        print('new model')
        model = Sequential()
        model.add(Dense(10, input_shape=(1,), activation='sigmoid', use_bias=True))
        model.add(Dense(1, use_bias=True))
        model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.RMSprop(0.1))
    else:
        model = load_model('model/' + str(region) + '/bikeinflow.h5')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
    model.fit(x=x, y=y, epochs=100, validation_split=0.1, verbose = 0, shuffle=True, batch_size=128,
              callbacks=[reduce_lr])
    model.save('model/' + str(region) + '/bikeinflow.h5')
    print(str(region) + '-bikeinflow:100 epoches have been trained over!')
    return
def train3(region):
    # path = 'model/' + str(region) + '/bikeoutflow.h5'
    # file = open(path, 'w')
    rawdata = pd.read_csv('model/' + str(region) + '/bikeoutflow.csv').values
    if len(rawdata) == 0:
        print('Region' + str(region) + ' don\'t have data about bikeoutflow!\nUse the unified model!')
    #     return
    dic = makedata(region, 'model/' + str(region) + '/bikeoutflow.csv')
    x = np.array(dic['x'])
    y = np.array(dic['y'])

    model = None
    if os.path.getsize('model/' + str(region) + '/bikeoutflow.h5') == 0:
        model = Sequential()
        model.add(Dense(10, input_shape=(1,),activation='sigmoid', use_bias=True))
        model.add(Dense(1, use_bias=True))
        model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.RMSprop(0.1))
    else:
        model = load_model('model/' + str(region) + '/bikeoutflow.h5')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
    model.fit(x=x, y=y, epochs=100, validation_split=0.1, verbose = 0, shuffle=True, batch_size=128,
              callbacks=[reduce_lr])
    model.save('model/' + str(region) + '/bikeoutflow.h5')
    print(str(region) + '-bikeoutflow:100 epoches have been trained over!')
    return
def train4(region):
    # path = 'model/' + str(region) + '/taxiinflow.h5'
    # file = open(path, 'w')
    rawdata = pd.read_csv('model/' + str(region) + '/taxiinflow.csv').values
    if len(rawdata) == 0:
        print('Region' + str(region) + ' don\'t have data about taxiinflow!\nUse the unified model!')
    #     return
    dic = makedata(region, 'model/' + str(region) + '/taxiinflow.csv')
    x = np.array(dic['x'])
    y = np.array(dic['y'])

    model = None
    if os.path.getsize('model/' + str(region) + '/taxiinflow.h5') == 0:
        model = Sequential()
        model.add(Dense(10, input_shape=(1,),activation='sigmoid', use_bias=True))
        model.add(Dense(1, use_bias=True))
        model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.RMSprop(0.1))
    else:
        model = load_model('model/' + str(region) + '/taxiinflow.h5')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
    model.fit(x=x, y=y, epochs=100, validation_split=0.1, verbose = 0, shuffle=True, batch_size=128,
              callbacks=[reduce_lr])
    model.save('model/' + str(region) + '/taxiinflow.h5')
    print(str(region) + '-taxiinflow:100 epoches have been trained over!')
    return
def train5(region):
    # path = 'model/' + str(region) + '/taxioutflow.h5'
    # file = open(path, 'w')
    rawdata = pd.read_csv('model/' + str(region) + '/taxioutflow.csv').values
    if len(rawdata) == 0:
        print('Region' + str(region) + ' don\'t have data about taxioutflow!\nUse the unified model!')
    #     return
    dic = makedata(region, 'model/' + str(region) + '/taxioutflow.csv')
    x = np.array(dic['x'])
    y = np.array(dic['y'])

    model = None
    if os.path.getsize('model/' + str(region) + '/taxioutflow.h5') == 0:
        model = Sequential()
        model.add(Dense(10, input_shape=(1,),activation='sigmoid', use_bias=True))
        model.add(Dense(1, use_bias=True))
        model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.RMSprop(0.1))
    else:
        model = load_model('model/' + str(region) + '/taxioutflow.h5')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
    model.fit(x=x, y=y, epochs=100, validation_split=0.1, verbose = 0, shuffle=True, batch_size=128,
              callbacks=[reduce_lr])
    model.save('model/' + str(region) + '/taxioutflow.h5')
    print(str(region) + '-taxioutflow:100 epoches have been trained over!')
    return

def train(region = 0,all = False):
    if all == True:
        for i in range(1,863):
            # train1(i)
            train2(i)
            train3(i)
            train4(i)
            train5(i)
        print('Have trained all the models in the dataset!')
        return
    # train1(region)
    train2(region)
    train3(region)
    train4(region)
    train5(region)
    print('Finished the task to train region' + str(region) + '!')
    return

'''
serve method
'''
def serve():
    demo = demonstrater()
    demo.run()
    return

'''
一个服务对象
'''
class demonstrater():
    def __init__(self):
        self.predictvaluestaxiinflow = [Queue() for x in range(862)]
        self.realvaluestaxiinflow = [Queue() for x in range(862)]
        self.predictvaluestaxioutflow = [Queue() for x in range(862)]
        self.realvaluestaxioutflow = [Queue() for x in range(862)]
        self.predictvaluesbikeinflow = [Queue() for x in range(862)]
        self.realvaluesbikeinflow = [Queue() for x in range(862)]
        self.predictvaluesbikeoutflow = [Queue() for x in range(862)]
        self.realvaluesbikeoutflow = [Queue() for x in range(862)]

        self.timestamp = 1388505600
        self.endtime = self.timestamp + 30 * 24 * 3600
        self.id = 0
        print('初始化完成!')
        return
    '''
    从时间戳获得日期
    日期格式统一为%Y-%m-%d %H:%M:%S
    '''
    def getdate(self):
        return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(self.timestamp))
    '''
    从日期获得时间戳
    日期格式统一为%Y-%m-%d %H:%M:%S
    '''
    def gettimestamp(self,date):
        return time.mktime(time.strptime(date,'%Y-%m-%d %H:%M:%S'))
    '''
    判断一个日期是否是属于当前slot的
    '''
    def judgedate(self,date,type):
        stamp = 0
        if type == 0:
            stamp = self.gettimestamp(self.formdate311(date))
        else:
            stamp = self.gettimestamp(self.formdateflow(date))
        if stamp > self.timestamp and stamp < self.timestamp + 1800:
            return True
        return False
    '''
    将311中的日期转换成标准格式
    '''
    def formdate311(self,date):
        add = False
        if date.__contains__('PM'):
            add = True
        date = date.replace(' AM','').replace(' PM','')
        stamp = time.mktime(time.strptime(date,'%m/%d/%Y %H:%M:%S'))
        if add == True:
            stamp += 12 * 3600
        return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(stamp))
    '''
    将flow文件中的日期转换成标准格式
    '''
    def formdateflow(self,date):
        '2014/6/1  0:07:00'
        if len(date.split(':')) == 2:
            date += ':00'
        stamp = time.mktime(time.strptime(date,'%Y/%m/%d  %H:%M:%S'))
        return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(stamp))

    '''
    将可使用的数据处理之后放在datacache文件夹中
    数据为从当前timestamp开始，三十分钟内的数据
    原始格式；
    311：<time,type,region>
    taxi、bike格式:<time,region,count>,分为inflow和outflow
    目标格式：
    311格式：<region,date,id,typelist>
    taxi、bike格式：<region,date,id,count>
    其中date是六数据的格式
    在进行数据处理的时候需要完成日期的转换
    '''
    def getdata(self):
        sdir = 'testdata/database/'
        ddir = 'testdata/datacache/'
        def getdata311():
            file = sdir + '311.csv'
            destination = ddir + '311.json'
            ans = {}
            for i in range(1, 863):
                ans[i] = []
            print('begin read!')
            data = pd.read_csv(file)[['time', 'region', 'type']]
            indexs = []
            length = len(data.values)
            print('数据长度：' + str(length))
            print('启动阶段批处理数据需要较长时间，请耐心等待！（如果需要观看程序运行结果，可以将数据集切至100条运行查看）')
            for i in range(length):
                if self.judgedate(data.values[i][0],0) == False:
                    indexs.append(i)
            data.drop(index=indexs)
            for i in range(length):
                ans[int(data.values[i][1])].append([data.values[i][0], data.values[i][2]])
            print('map over!')
            def reduce(lis):
                for i in range(len(lis)):
                    time = lis[i][0].split(' ')
                    date = time[0]
                    a = time[1].split(':')
                    x = int(a[0])
                    y = int(a[1])
                    b = time[2]
                    id = 2 * x + int(y / 30)
                    if b.__contains__('PM'):
                        id = id + 24
                    lis[i] = [date, id, lis[i][1]]
                ret = {}
                for entry in lis:
                    if ret.__contains__((entry[0], entry[1])):
                        ret[(entry[0], entry[1])].append(entry[2])
                    else:
                        ret[(entry[0], entry[1])] = [entry[2]]
                ending = []
                for key in ret.keys():
                    ending.append([key[0], key[1], ret[key]])
                return ending
            ret = {}
            print('begin reduce!')
            for i in range(1, 863):
                ret[i] = reduce(ans[i])
            print('311 reduce over!')
            f = open('testdata/datacache/311.json','w')
            ret = json.dumps(ret)
            # print(ret)
            f.write(ret)
            f.close()
            return
        def getdataflow():
            filelist = [sdir + 'bikeinflow.csv', sdir + 'bikeoutflow.csv', sdir + 'taxiinflow.csv',
                        sdir + 'taxioutflow.csv']
            destination = [ddir + 'bikeinflow.json',ddir + 'bikeoutflow.json',ddir + 'taxiinflow.json',ddir + 'taxioutflow.json']
            ans = {}

            def init():
                for i in range(1, 863):
                    ans[i] = []
                return

            def map(lis):
                for entry in lis:
                    ans[int(entry[1])].append([entry[0], entry[2]])
                return

            def reduce(lis):
                length = len(lis)
                for i in range(length):
                    entry = lis[i][0].split(' ')
                    lis[i] = [entry[0], entry[1], lis[i][1]]
                for i in range(length):
                    entry = lis[i][1].split(':')
                    x = int(entry[0])
                    y = int(entry[1])
                    lis[i][1] = 2 * x + int(y / 30)
                ret = {}
                for entry in lis:
                    if ret.__contains__((entry[0], entry[1])):
                        ret[(entry[0], entry[1])] = ret[(entry[0], entry[1])] + int(entry[2])
                    else:
                        ret[(entry[0], entry[1])] = int(entry[2])
                ending = []
                for key in ret.keys():
                    ending.append([key[0], key[1], ret[key]])
                if len(ending) == 0:
                    return 0
                else:
                    return ending[0][2]

            for j in range(4):
                data = pd.read_csv(filelist[j])
                indexs = []
                length = len(data.values)
                for i in range(length):
                    if self.judgedate(data.values[i][0],1) == False:
                        indexs.append(i)
                data.drop(index=indexs)
                init()
                map(data.values)
                file = open(destination[j],'w')

                ret = {}
                for i in range(1, 863):
                    ret[i] = reduce(ans[i])
                file = open(destination[j], 'w')
                ret = json.dumps(ret)
                file.write(ret)
                file.close()
                print(destination[j] + ' over!')
            return

        #311
        print('开始获取311数据集的信息')
        getdata311()
        print('311信息获取完毕！')
        #taxi
        #bike
        print('dataflow')
        getdataflow()
        print('flow over!')
        return
    '''
    计算异常度，将结果保存在队列中，对队列进行更新
    '''
    def process(self):
        ags = {}
        file = open('testdata/datacache/311.json','r')
        data311 = json.load(file)
        file.close()
        file = open('testdata/datacache/taxiinflow.json','r')
        datataxiinflow = json.load(file)
        file.close()
        file = open('testdata/datacache/taxioutflow.json', 'r')
        datataxioutflow = json.load(file)
        file.close()
        file = open('testdata/datacache/bikeinflow.json', 'r')
        databikeinflow = json.load(file)
        file.close()
        file = open('testdata/datacache/bikeoutflow.json', 'r')
        databikeoutflow = json.load(file)
        file.close()
        #先计算没有跨越地区的异常度
        for i in range(1,863):
            ag311 = 1
            if data311[str(i)] == '0':
                ag311 = 0
            print(datataxiinflow[str(i)])
            self.realvaluestaxiinflow[i - 1].put(int(datataxiinflow[str(i)]))
            if self.realvaluestaxiinflow[i - 1].qsize() > 48:
                self.realvaluestaxiinflow[i - 1].get()
            self.predictvaluestaxiinflow[i - 1].put(predict(i, 'taxiinflow', self.id))
            if self.predictvaluestaxiinflow[i - 1].qsize() > 48:
                self.predictvaluestaxiinflow[i - 1].get()
            ag3 = adapteddeviation(self.realvaluestaxiinflow[i - 1],self.predictvaluestaxiinflow[i - 1])

            self.realvaluestaxioutflow[i - 1].put(int(datataxioutflow[str(i)]))
            if self.realvaluestaxioutflow[i - 1].qsize() > 48:
                self.realvaluestaxioutflow[i - 1].get()
            self.predictvaluestaxioutflow[i - 1].put(predict(i, 'taxioutflow', self.id))
            if self.predictvaluestaxioutflow[i - 1].qsize() > 48:
                self.predictvaluestaxioutflow[i - 1].get()
            ag4 = adapteddeviation(self.realvaluestaxioutflow[i - 1], self.predictvaluestaxioutflow[i - 1])

            self.realvaluesbikeinflow[i - 1].put(int(databikeinflow[str(i)]))
            if self.realvaluesbikeinflow[i - 1].qsize() > 48:
                self.realvaluesbikeinflow[i - 1].get()
            self.predictvaluesbikeinflow[i - 1].put(predict(i, 'bikeinflow', self.id))
            if self.predictvaluesbikeinflow[i - 1].qsize() > 48:
                self.predictvaluesbikeinflow[i - 1].get()
            ag1 = adapteddeviation(self.realvaluesbikeinflow[i - 1], self.predictvaluesbikeinflow[i - 1])

            self.realvaluesbikeoutflow[i - 1].put(int(databikeoutflow[str(i)]))
            if self.realvaluesbikeoutflow[i - 1].qsize() > 48:
                self.realvaluesbikeoutflow[i - 1].get()
            self.predictvaluesbikeoutflow[i - 1].put(predict(i, 'bikeoutflow', self.id))
            if self.predictvaluesbikeoutflow[i - 1].qsize() > 48:
                self.predictvaluesbikeoutflow[i - 1].get()
            ag2 = adapteddeviation(self.realvaluesbikeoutflow[i - 1], self.predictvaluesbikeoutflow[i - 1])
            localags = [ag1,ag2,ag3,ag4,ag311]

            rag = calWholeAg(localags)
            ags[i] = rag
        #计算跨越地区的异常度
        ags = implementag(ags)
        return ags
    '''
    判断是否能输出
    '''
    def canoutput(self):
        if self.realvaluestaxiinflow[0].qsize() == 48:
            return True
        return False
    '''
    将异常检测结果写到report.csv中
    '''
    def report(self,ags):
        columns = ['region','time']
        data = pd.read_csv('report.csv').values
        for key in ags.keys():
            if ags[key] >= ANOMALY:
                data.append([key,self.timestamp])
        df = pd.DataFrame(data,columns = columns)
        df.to_csv('report.csv',index = False)
        return
    '''
    判断读取数据是否结束
    如果所有数据集中的数据都没有数据了就结束
    统一时间一个月
    '''
    def judgeending(self):
        if self.timestamp >= self.endtime:
            return True
        return False
    def run(self):
        while True:
            if self.judgeending():
                break
            print('judge over!')
            starttime = time.time()
            print('tmp time is ' + self.getdate())
            self.getdata()
            print('数据获取完成！')
            # time.sleep(10000)
            ags = self.process()
            print('异常处理完成！')
            if self.canoutput():
                self.report(ags)
                print('数据报告结束！')
            self.timestamp += 1800
            self.id = (self.id + 1) % 48
            endtime = time.time()
            print('one round over!')
            print()
            time.sleep(int(100 + starttime - endtime))
        return
    def __del__(self):
        return
# base = 398
# import sys
# id = sys.argv[1]
# id = int(id)
# for i in range(base + id * 5,base + id * 5 + 5):
#     train(i)
# for i in range(498,507):
#     train(i)
# q = Queue()
# q.put(1)
# q.put(2)
# q.put(3)
# print(q.full())
# def func(Q):
#     Q.put(4)
#     return
# qq = q
# qq.put(4)
# print(q.qsize())
serve()