# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import re

#logfile = "logsfromAli/mouthreglog/1104-1501.txt"
logfile = "logsfromAli/mouthreglog/1029-1006.txt"
with open(logfile) as f:
    data = f.read()

pattern = re.compile('solver.cpp:243] Iteration (\d+)')
results = re.findall(pattern, data)
iter_num = []
for result in results:
    iter_num.append(int(result))


pattern = re.compile('Train net output #0: RegressionLoss = ([\.\deE+-]+)')
results = re.findall(pattern, data)
mbox_loss = []
for result in results:
    regloss = float(result)
    mbox_loss.append(regloss)
#    if regloss < 0.1 :
#        mbox_loss.append(regloss)
#    else:
#        iter_num.pop()

#print mbox_loss
pattern = re.compile(', lr = ([\.\deE+-]+)')
results = re.findall(pattern, data)
learning_rate = []
for result in results:
    learning_rate.append(float(result))
#print learning_rate
short = min(len(iter_num), len(mbox_loss), len(learning_rate))
print len(iter_num), len(mbox_loss), len(learning_rate), short

pattern = re.compile('solver.cpp:358] Iteration (\d+)')
results = re.findall(pattern, data)
testiter_num = []
for result in results:
    testiter_num.append(int(result))
pattern = re.compile('Test net output #0: RegressionLoss = ([\.\deE+-]+)')
results = re.findall(pattern, data)
testreg_loss = []
for result in results:
    testreg_loss.append(float(result))


plt.subplot(211)
plt.title(logfile)
plt.grid()
plt.plot(iter_num[:short], mbox_loss[:short],'.')
#plt.ylim(0.001,1.5)  #联合训练，训分类时回归loss会为0， 不画在图上
plt.plot(testiter_num[:len(testreg_loss)], testreg_loss[:len(testreg_loss)],'r*')
plt.subplot(212)
plt.grid() 
plt.plot(iter_num[:short], learning_rate[:short])
plt.show()
