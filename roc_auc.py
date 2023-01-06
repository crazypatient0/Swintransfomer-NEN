import csv
import time

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

thlist = [0.9, 0.8, 0.7, 0.6, 0.5, .04, 0.3, 0.2, 0.1]
with open('maps/swinv2-tiny-16-256-rgb-ep81-pd2/csv_file.csv', "r", encoding='UTF-8') as f:
    data = csv.reader(f)
    datalist = []
    for row in data:
        datalist.append(row)
print(datalist)

ppl = []
tprl = []
fprl = []
for i in thlist:
    print(i)
    tp = 0
    fp = 0
    for j in datalist:
        prob = j[2]
        if float(prob) >= i:
            name = j[0]
            label = j[1]
            if label == 'Tumer1N1' and 'Tumer1N1' in name:
                tp += 1
            if 'Tumer1N0' in name:
                fp +=1
    print(tp,fp)
    tpr = tp/1000
    fpr = fp/1000
    print(fpr,tpr)
    tprl.append(tpr)
    fprl.append(fpr)
    print('-----------')

plt.plot(fprl, tprl, label='ROC')

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()