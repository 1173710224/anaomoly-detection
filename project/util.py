'''
cbz
those libraries used
'''
import os
import pandas as pd
import json
from math import sqrt
from keras.models import load_model
CONTRIBUTION = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625,
                0.0078125, 0.00390625, 0.001953125, 0.0009765625,
                0.00048828125, 0.000244140625, 0.0001220703125,
                6.103515625e-05, 3.0517578125e-05, 1.52587890625e-05,
                7.62939453125e-06, 3.814697265625e-06, 1.9073486328125e-06,
                9.5367431640625e-07, 4.76837158203125e-07, 2.384185791015625e-07,
                1.1920928955078125e-07, 5.960464477539063e-08, 2.9802322387695312e-08,
                1.4901161193847656e-08, 7.450580596923828e-09, 3.725290298461914e-09,
                1.862645149230957e-09, 9.313225746154785e-10, 4.656612873077393e-10,
                2.3283064365386963e-10, 1.1641532182693481e-10, 5.820766091346741e-11,
                2.9103830456733704e-11, 1.4551915228366852e-11, 7.275957614183426e-12,
                3.637978807091713e-12, 1.8189894035458565e-12, 9.094947017729282e-13,
                4.547473508864641e-13, 2.2737367544323206e-13, 1.1368683772161603e-13,
                5.684341886080802e-14, 2.842170943040401e-14, 1.4210854715202004e-14,
                7.105427357601002e-15,7.105427357601002e-15]
ANOMALY = 1
'''
判断文件是否为空
'''
def isEmpty(file):
    if os.path.getsize(file) == 0:
        return True
    return False

# file = open('test.h5','w')
# file.close()
def calWholeAg(ags):
    configure = getConfigure()
    weight = configure['weight']
    m = int(configure['m'])
    ret = 0
    for i in range(m):
        ret = ret + pow(ags[i] * weight[i],2)
    return sqrt(ret / m)
def setConfigure(weight = None,m = None,ag311 = None,threshold = None):
    data = getConfigure()
    if weight != None:
        data['weight'] = weight
    if m != None:
        data['m'] = m
    if ag311 != None:
        data['ag311'] = ag311
    if threshold != None:
        data['threshold'] = threshold
    data = json.dumps(data)
    file = open('configure.json', 'w')
    file.write(data)
    file.close()
    return
def getConfigure():
    file = open('configure.json', 'r')
    data = json.load(file)
    file.close()
    return data
'''
从anomaly.csv文件中获取异常数据
'''
def getAnomaly():
    data = pd.read_csv('anomaly.csv')[['id','ag1','ag2','ag3','ag4','label']].values
    return data
'''
返回当前参数下预测的错误率
先将参数在文件中进行设置
从文件中读取数据
计算预测值，同时统计错误率
'''
def errorrate(w0,w1,w2,w3,w4,ag311,
              threshold0,threshold1,threshold2,threshold3,threshold4,threshold5,
              threshold6,threshold7,threshold8,threshold9,threshold10,threshold11,
              threshold12,threshold13,threshold14,threshold15,threshold16,threshold17,
              threshold18,threshold19,threshold20,threshold21,threshold22,threshold23,
              threshold24,threshold25,threshold26,threshold27,threshold28,threshold29,
              threshold30,threshold31,threshold32,threshold33,threshold34,threshold35,
              threshold36,threshold37,threshold38,threshold39,threshold40,threshold41,
              threshold42,threshold43,threshold44,threshold45,threshold46,threshold47):
    weight = []
    for i in range(5):
        weight.append(eval('w' + str(i)))
    threshold = []
    for i in range(48):
        threshold.append(eval('threshold' + str(i)))
    setConfigure(weight=weight,threshold=threshold)
    traindata = getAnomaly()
    errornum = 0
    num = len(traindata)
    for entry in traindata:
        ags = [entry[1],entry[2],entry[3],entry[4],ag311]
        tag = calWholeAg(ags)
        anomaly = 0
        if tag >= eval('threshold' + str(entry[0])):
            anomaly = 1
        if anomaly != int(entry[5]):
            errornum = errornum + 1
    return errornum / num
# setConfigure()
# lis = {}
# for i in range(48):
#     lis[i] = 1
# setConfigure(threshold=lis)
# setConfigure(ag311 = 1)
'''
predict method
对一定的时间进行预测
'''
def predict(region,filename,id):
    path = 'model/' + str(region) + '/' + filename + '.h5'
    model = None
    if os.path.getsize(path) == 0:
        model = load_model('model/' + filename + '.h5')
    else:
        model = load_model(path)
    ans = model.predict([id])
    return ans[0]
'''
计算所获得数据的偏离度
计算结果就是一个单独的ag
'''
def deviation(realvalues,predictvalues):
    ret = 0
    print(len(CONTRIBUTION))
    print(len(realvalues))
    print(len(predictvalues))
    for i in range(min(len(realvalues),48)):
        if realvalues[i] == 0:
            ret += 0
            continue
        ret += CONTRIBUTION[47 - i] * abs(realvalues[i] - predictvalues[i]) / realvalues[i]
    ret /= min(len(realvalues),48)
    return ret
def adapteddeviation(q1,q2):
    l1 = []
    l2 = []
    while q1.empty() == False:
        l1.append(q1.get())
    while q2.empty() == False:
        l2.append(q2.get())
    for item in l1:
        q1.put(item)
    for item in l2:
        q2.put(item)
    return deviation(l1,l2)
'''
添加地区之间相互影响
'''
def implementag(dic):
    file = open('adjacent.json','r')
    data = json.load(file)
    ret = {}
    for i in range(1,863):
        adjacents = data[str(i)]
        ans = dic[i] * 0.5
        epsilon = 0.5 / len(adjacents)
        for region in adjacents:
            ans += dic[int(region)] * epsilon
        ret[i] = ans
    return ret

def getAdjacents(id):
    file = open('adjacent.json', 'r')
    data = json.load(file)
    file.close()
    return data[int(id)]
'''
将异常区域进行合并从而实现分类
'''

def classify(regions):
    ans = [0 for x in range(863)]
    def find(id):
        while ans[id] > 0:
            id = ans[id]
        return id
    for x in regions:
        ans[x] = x
    for x in regions:
        lis = getAdjacents(int(x))
        for tmp in lis:
            ans[tmp] = find(x)
    ret = {}
    for x in range(863):
        ret[x] = []
    for x in range(863):
        ret[ans[x]].append(x)
    return ret
'''
判断该时间段是否异常
'''
def abnormal(stamp):
    # import time
    # ablist = []
    # timelist = []
    # timelist.append(['2014-10-31 21:00:00','2014-11-1 2:00:00'])
    # timelist.append(['2014-10-31 7:00:00','2014-11-1 3:00:00'])
    # timelist.append(['2014-11-5 10:30:00','2014-11-7 17:45:00'])
    # timelist.append(['2014-11-5 18:00:00','2014-11-9 23:00:00'])
    # timelist.append(['2014-11-6 20:00:00','2014-11-9 23:00:00'])
    # timelist.append(['2014-11-7 18:00:00','2014-11-7 22:00:00'])
    # timelist.append(['2014-11-7 19:00:00','2014-11-8 21:00:00'])
    # timelist.append(['2014-11-11 10:00:00','2014-11-15 20:00:00'])
    # timelist.append(['2014-11-13 9:00:00','2014-11-15 6:00:00'])
    # timelist.append(['2014-11-15 9:30:00','2014-11-19 18:30:00'])
    # timelist.append(['2014-11-15 13:00:00','2014-11-15 16:00:00'])
    # timelist.append(['2014-11-15 14:00:00','2014-11-15 18:00:00'])
    # timelist.append(['2014-11-17 18:00:00','2014-11-17 21:00:00'])
    # timelist.append(['2014-11-18 11:00:00','2014-11-20 20:00:00'])
    # # time.mktime(time.strptime(date, '%Y-%m-%d %H:%M:%S'))
    # for entry in timelist:
    #     ablist.append([time.mktime(time.strptime(entry[0], '%Y-%m-%d %H:%M:%S')),time.mktime(time.strptime(entry[1], '%Y-%m-%d %H:%M:%S'))])
    # print(ablist)
    ablist = [[1414760400.0, 1414778400.0], [1414710000.0, 1414782000.0], [1415154600.0, 1415353500.0], [1415181600.0, 1415545200.0], [1415275200.0, 1415545200.0], [1415354400.0, 1415368800.0], [1415358000.0, 1415451600.0], [1415671200.0, 1416052800.0], [1415840400.0, 1416002400.0], [1416015000.0, 1416393000.0], [1416027600.0, 1416038400.0], [1416031200.0, 1416045600.0], [1416218400.0, 1416229200.0], [1416279600.0, 1416484800.0]]
    for entry in ablist:
        if stamp > entry[0] and stamp < entry[1]:
            return True
    return False

# abnormal(1)