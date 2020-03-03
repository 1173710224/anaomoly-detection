'''
author:
cbz
'''
import pandas as pd
from functools import cmp_to_key
from hyperopt import hp,fmin,rand,tpe
from functools import partial
from util import *
import os
'''
mapreduce method
construct the training source data for 311
'''
def mapreduce311():
    file = 'data/311/311.csv'
    destination = '311.csv'
    ans = {}
    for i in range(1,863):
        ans[i] = []
    data = pd.read_csv(file)[['time','region','type']].values
    length = len(data)
    for i in range(length):
        ans[int(data[i][1])].append([data[i][0],data[i][2]])
    def reduce(lis):
        for i in range(len(lis)):
            time = lis[i][0].split(' ')
            date = time[0]
            a = time[1].split(':')
            x = int(a[0])
            y = int(a[1])
            b = time[2]
            id = 2 * x + int(y / 30)
            if b.__contains__('P'):
                id  = id + 24
            lis[i] = [date,id,lis[i][1]]
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
    for i in range(1,863):
        data = reduce(ans[i])
        df = pd.DataFrame(data,columns = ['date','id','types'])
        df.to_csv('model/' + str(i) + '/' + destination,index = False)
    return
'''
mapreduce method
construct the training source data for flow
'''
def mapreduceflow():
    filelist = ['data/bike/bikeinflow.csv','data/bike/bikeoutflow.csv','data/taxi/taxiinflow.csv','data/taxi/taxioutflow.csv']
    destination = ['bikeinflow.csv','bikeoutflow.csv','taxiinflow.csv','taxioutflow.csv']
    ans = {}
    def init():
        for i in range(1, 863):
            ans[i] = []
        return
    def map(lis):
        for entry in lis:
            ans[int(entry[1])].append([entry[0],entry[2]])
        return
    def reduce(lis):
        length = len(lis)
        for i in range(length):
            entry = lis[i][0].split(' ')
            lis[i] = [entry[0],entry[1],lis[i][1]]
        for i in range(length):
            entry = lis[i][1].split(':')
            x = int(entry[0])
            y = int(entry[1])
            lis[i][1] = 2 * x + int(y / 30)
        ret = {}
        for entry in lis:
            if ret.__contains__((entry[0],entry[1])):
                ret[(entry[0],entry[1])] = ret[(entry[0],entry[1])] + int(entry[2])
            else:
                ret[(entry[0], entry[1])] = int(entry[2])
        ending = []
        for key in ret.keys():
            ending.append([key[0],key[1],ret[key]])
        return ending
    for j in range(4):
        data = pd.read_csv(filelist[j])
        init()
        map(data.values)
        for i in range(1 , 863):
            data = reduce(ans[i])
            df = pd.DataFrame(data,columns=['date','id','count'])
            df.to_csv('model/' + str(i) + '/' + destination[j],index = False)
    return
'''
makedata method
constuct the direct training data used to train the model for a designated region.
return a dictrionary containing x and y
'''
def makedata(region,filename):
    if filename.__contains__('311.csv'):
        print('311 file has no distribution,just anomaly.')
        return
    data = pd.read_csv(filename)[['id','count']].values
    ans = {}
    for entry in data:
        if ans.__contains__(int(entry[0])):
            ans[int(entry[0])].append(int(entry[1]))
        else:
            ans[int(entry[0])] = [int(entry[1])]
    for entry in ans.keys():
        tmp = ans[entry]
        sum = 0
        for item in tmp:
            sum += item
        ans[entry] = sum / len(tmp)
    ret = {}
    for i in range(48):
        if ans.__contains__(i) == False:
            ans[i] = 0
    ret['x'] = []
    ret['y'] = []
    for key in ans.keys():
        ret['x'].append([key])
        ret['y'].append([ans[key]])
    return ret
# mapreduceflow()
# mapreduce311()
'''
创建目录结构：.h5文件
'''
def makestructure():
    filelist = ['311.h5','bikeinflow.h5','bikeoutflow.h5','taxiinflow.h5','taxioutflow.h5']
    for i in range(1,863):
        path = 'model/' + str(i) + '/'
        for file in filelist:
            f = open(path + file,'w')
            f.close()
    return
'''
创建nn模型保存文件
'''
def make2():
    filelist = ['311.h5', 'bikeinflow.h5', 'bikeoutflow.h5', 'taxiinflow.h5', 'taxioutflow.h5']
    path = 'model/'
    for file in filelist:
        f = open(path + file, 'w')
        f.close()
    return
'''
获取所有的一级数据文件名以及相应路径
'''
def getdatapath():
    filelist = ['311.csv','bikeinflow.csv','bikeoutflow.csv','taxiinflow.csv','taxioutflow.csv']
    ans = []
    for i in range(1,863):
        path = 'model/' + str(i) + '/'
        for j in range(5):
            ans.append(path + filelist[j])
    return ans
'''
将所有的数据文件进行排序
一级排序：date
二级排序：id
'''
def comdate311(d1,d2):
    d1 = d1.split('/')
    d2 = d2.split('/')
    if int(d1[2]) > int(d2[2]):
        return 1
    if int(d1[2]) < int(d2[2]):
        return -1
    if int(d1[0]) > int(d2[0]):
        return 1
    if int(d1[0]) < int(d2[0]):
        return -1
    if int(d1[1]) > int(d2[1]):
        return 1
    if int(d1[1]) < int(d2[1]):
        return -1
    return 0
def comdateflow(d1,d2):
    d1 = d1.split('/')
    d2 = d2.split('/')
    if int(d1[0]) > int(d2[0]):
        return 1
    if int(d1[0]) < int(d2[0]):
        return -1
    if int(d1[1]) > int(d2[1]):
        return 1
    if int(d1[1]) < int(d2[1]):
        return -1
    if int(d1[2]) > int(d2[2]):
        return 1
    if int(d1[2]) < int(d2[2]):
        return -1
    return
def compare(t1,t2):
    #判断是不是311文件
    tmp = t1[0].split('/')
    if len(tmp[0]) == 4:
        if comdateflow(t1[0],t2[0]) > 0:
            return 1
        elif comdateflow(t1[0],t2[0]) < 0:
            return -1
        else:
            return int(t1[1]) - int(t2[1])
    else:
        if comdate311(t1[0],t2[0]) > 0:
            return 1
        elif comdate311(t1[0],t2[0]) < 0:
            return -1
        else:
            return int(t1[1]) - int(t2[1])
def sortsingle(file):
    data = pd.read_csv(file)
    columns = data.columns
    data = data.values
    data = sorted(data, key=cmp_to_key(compare))
    data = pd.DataFrame(data, columns=columns)
    data.to_csv(file, index=False)
    return
def sort():
    filelist = getdatapath()
    for file in filelist:
        print(file)
        sortsingle(file)
        print('File ' + file + ' has been processed over!')
    return
# make2()
# makestructure()
# sortsingle('model/12/311.csv')
# sort()
# d1 = ['01/01/2015',1]
# d2 = ['01/02/2014',2]
# print(comdate311(d1,d2))
'''
构造训练数据
数据用于确定模型中的超参数
将结果保存在anomaly.csv中<id,ag1,ag2,ag3,ag4,ag5,label>
'''
def makeanomaly():

    return
'''
计算各个地区的邻接地区
将结果保存在adjacent.json
'''
def adjacentarea():
    file = open('data/region.txt','r')
    data = file.readlines()
    file.close()
    length = len(data)
    for i in range(length):
        data[i] = data[i].replace('\n','').split('\t')
    ans = {}
    dir = [[1,1],[1,0],[1,-1],[0,1],[0,-1],[-1,1],[-1,0],[-1,-1]]
    for i in range(len(data)):
        for j in range(len(data[i])):
            if int(data[i][j]) == 0:
                continue
            value = int(data[i][j])
            for k in range(8):
                if int(data[i + dir[k][0]][j + dir[k][1]]) == 0:
                    continue
                if value != int(data[i + dir[k][0]][j + dir[k][1]]):
                    if ans.__contains__(value):
                        if ans[value].__contains__(int(data[i + dir[k][0]][j + dir[k][1]])) == False:
                            ans[value].append(int(data[i + dir[k][0]][j + dir[k][1]]))
                    else:
                        ans[value] = [int(data[i + dir[k][0]][j + dir[k][1]])]
    file = open('adjacent.json','w')
    dic = json.dumps(ans)
    file.write(dic)
    file.close()
    return

'''
训练模型中的超参数
训练结束之后将结果写入配置文件中
'''
def trainParam():
    space = []
    for i in range(5):
        space.append(hp.uniform('w' + str(i),0,1))
    space.append(hp.uniform('ag311',0,1))
    for i in range(48):
        space.append(hp.uniform('threshold' + str(i), 0, 1))
    algo = partial(tpe.suggest, n_startup_jobs=10)
    best = fmin(errorrate, space, algo=algo,max_evals=500)
    print(best)
    setConfigure(weight=[best['w0'],best['w1'],best['w2'],best['w3'],0.23])
    setConfigure(ag311=best['ag311'])
    setConfigure(threshold=[best['threshold' + str(i)] for i in range(48)])
    return
def makeparams():
    lis = ''
    for i in range(5):
        lis = lis + ('w' + str(i) + ',')
    lis = lis + ('ag311,')
    for i in range(48):
        lis = lis + ('threshold' + str(i) + ',')
    print(lis)
# adjacentarea()
'''
手动生成实验数据
311格式：<time,type,latitude,longitude>
bike格式：<starttime,startstationlatitude,startstationlongitude,endstationlatitude,endstationlatitude>
taxi格式：<pickuptime,pickuplatitude,pickuplongitude,dropofftime,dropofflatitude,dropofflongitude,count>
'''
def maketestdata():
    return
# import time
# print(time.mktime(time.strptime('2014-01-01 00:00:00',"%Y-%m-%d %H:%M:%S")))
# print(time.localtime())
# trainParam()