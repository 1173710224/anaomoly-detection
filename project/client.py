'''
author:
cbz
'''
from core import *
print('Welcome!')
print('Would you like to continue training your model or begin to run service?')
print('If the former, please input 1,otherwise input 2.')
choice = input()
if choice == 1:
    print('Would you like to train all the regions or a designated region?')
    choice = input('If the former,please input 1,otherwise 2.')
    if choice == 1:
        train(all = True)
    else:
        id = input('Please input the region id!')
        train(region = id)
else:
    serve()
    print('Anomaly has been in the report.csv, please check it!')
print('one round over!')
while True:
    print('Would you like to continue training your model or begin to run service?')
    print('If the former, please input 1,otherwise input 2.')
    choice = input()
    if choice == 1:
        print('Would you like to train all the regions or a designated region?')
        choice = input('If the former,please input 1,otherwise 2.')
        if choice == 1:
            train(all=True)
        else:
            id = input('Please input the region id!')
            train(region=id)
    else:
        serve()
        print('Anomaly has been in the report.csv, please check it!')
    print('one round over!')