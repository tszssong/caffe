# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import re
logfilelist = [
               "logsfromAli/1120-0822.txt", \
               "logsfromAli/2018-11-19_14-17-50.log"
               ]

def get_lr_loss( logfile ):
    
    with open(logfile) as f:
        data = f.read()

    pattern = re.compile('solver.cpp:243] Iteration (\d+)')
    results = re.findall(pattern, data)
    iter_num = []
    for result in results:
        iter_num.append(int(result))

    pattern = re.compile('Train net output #0: RegressionLoss = ([\.\deE+-]+)')
    results = re.findall(pattern, data)
    train_reg_loss = []
    for result in results:
        regloss = float(result)
        train_reg_loss.append(regloss)

    pattern = re.compile('solver.cpp:358] Iteration (\d+)')
    results = re.findall(pattern, data)
    test_iter_num = []
    for result in results:
        test_iter_num.append(int(result))

    pattern = re.compile('Test net output #0: RegressionLoss = ([\.\deE+-]+)')
    results = re.findall(pattern, data)
    test_reg_loss = []
    for result in results:
        test_reg_loss.append(float(result))

    pattern = re.compile(', lr = ([\.\deE+-]+)')
    results = re.findall(pattern, data)
    learning_rate = []
    for result in results:
        learning_rate.append(float(result))
    #print learning_rate
    short = min(len(iter_num), len(train_reg_loss), len(learning_rate))
    print len(iter_num), len(train_reg_loss), len(learning_rate), short
    return iter_num[:short], train_reg_loss[:short], learning_rate[:short],test_iter_num,test_reg_loss
colorlists = ['red','blue','green','black','pink','darkorange','cyan']
linestylelists = ['-', '--', '-.', ':', '-', '--', '-.', ':']
def drawlogs(dict_train_num_, dict_train_loss_, dict_test_num_, dict_test_loss_, dict_lr_):
    plt.subplot(211)
#    plt.title(logfile)
    plt.grid()
    for i in xrange(len(dict_train_num_)):
        plt.plot(dict_train_num_[i][:len(dict_train_loss_[i])], \
                 dict_train_loss_[i][:len(dict_train_loss_[i])], linewidth = '1', color=colorlists[i], label = "train"+str(i))
        plt.ylim(0.001,1.0)  #联合训练，训分类时回归loss会为0， 不画在图上
        plt.plot(dict_test_num_[i][:], \
                 dict_test_loss_[i][:], \
                 color=colorlists[i], linestyle=linestylelists[i], marker='*' )  #label = "test"+str(i)
    plt.legend(loc='upper left')
    plt.subplot(212)
    plt.grid()
#    plt.ylim(0,0.0001)
    for i in xrange(len(dict_train_num_)):
        plt.plot(dict_train_num_[i][:len(dict_train_loss_[i])], \
                 dict_lr_[i][:len(dict_train_loss_[i])], \
                 color=colorlists[i],linestyle=linestylelists[i] )
    plt.show()

dict_Iter_train = {}
dict_Loss_train = {}
dict_Iter_test = {}
dict_Loss_test = {}
dict_LR = {}
for i in xrange( len(logfilelist) ):
    dict_Iter_train[i] = []
    dict_Loss_train[i] = []
    dict_Iter_test[i] = []
    dict_Loss_test[i] = []
    dict_LR[i] = []
    trainIterNum, train_loss, lr,testIterNum,test_loss = get_lr_loss( logfilelist[i] )
    dict_Iter_train[i] = trainIterNum
    dict_Loss_train[i] = train_loss
    dict_Iter_test[i] = testIterNum
    dict_Loss_test[i] = test_loss
    dict_LR[i] = lr
    print len(trainIterNum),len(testIterNum), len(test_loss)
print len(dict_Iter_test), len(dict_Iter_train)
drawlogs(dict_Iter_train, \
         dict_Loss_train, \
         dict_Iter_test, \
         dict_Loss_test, \
         dict_LR, \
         )
