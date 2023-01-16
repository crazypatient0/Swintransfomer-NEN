import json
import os.path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable



class Nen(nn.Module):  # 定义网络
    def __init__(self):
        super(Nen, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.w1 = Variable(torch.tensor(np.random.rand(1)))
        self.w2 = 1-self.w1
        self.ll = torch.unsqueeze(torch.tensor((self.w1,self.w2)),1)

    def forward(self, x):
        x = self.calc_neightbor_info(x)  # (9,2)

        # x = self.linear(x)
        x= self.multi(x)
        x = self.sigmoid(x)
        x = self.calc_neightbor_info(x)
        # x = self.linear(x)
        x = self.multi(x)
        x = self.sigmoid(x)
        x = self.calc_neightbor_info(x)
        # x = self.linear(x)
        x = self.multi(x)
        x = self.sigmoid(x)
        x = self.nto(x)
        return x

    # input(9,1) calc output(9,2)
    def calc_neightbor_info(self, x):
        x_list = []
        # print(x)
        # time.sleep(10)
        # print('cnix',len(x))
        for i in x:
            temp = i
            # print(i)
            a00 = round(float(temp[0]), 4)
            a01 = round(float(temp[1]), 4)
            a02 = round(float(temp[2]), 4)
            a10 = round(float(temp[3]), 4)
            a11 = round(float(temp[4]), 4)
            a12 = round(float(temp[5]), 4)
            a20 = round(float(temp[6]), 4)
            a21 = round(float(temp[7]), 4)
            a22 = round(float(temp[8]), 4)
            b00 = round(((a01 + a10 + a11) / 3), 4)
            b01 = round(((a00 + a10 + a11 + a12 + a02) / 5), 4)
            b02 = round(((a01 + a11 + a12) / 3), 4)
            b10 = round(((a00 + a01 + a11 + a21 + a20) / 5), 4)
            b11 = round(((a00 + a01 + a02 + a10 + a12 + a20 + a21 + a22) / 8), 4)
            b12 = round(((a02 + a01 + a11 + a21 + a22) / 5), 4)
            b20 = round(((a21 + a11 + a10) / 3), 4)
            b21 = round(((a20 + a10 + a11 + a12 + a22) / 5), 4)
            b22 = round(((a21 + a11 + a12) / 3), 4)
            a = np.array([
                [a00, b00],
                [a01, b01],
                [a02, b02],
                [a10, b10],
                [a11, b11],
                [a12, b12],
                [a20, b20],
                [a21, b21],
                [a22, b22]])
            x_list.append(a)
        x_list = torch.tensor(np.array(x_list), dtype=torch.float32)
        return x_list

    # from(9,1) to (1,1)after conv
    def nto(self, x):
        tri = 0
        re=''
        for i in x:
            a = i[0, :]
            # a = torch.unsqueeze(a, 0)
            if tri>0:
                re = torch.cat((re, a), 0)
                tri+=1
            else:
                re = a
                tri+=1
            # print(a)
            # print(a.shape)
            # print(re_list)
            # time.sleep(10)
        return re

    # test w1 w 2
    def multi(self,x):
        # print(x)
        x_list = []
        for i in range(len(x)):
            x[i] = x[i].to(torch.float32)
            self.ll = self.ll.to(torch.float32)
            temp = torch.mm(x[i], self.ll)
            a= np.array([
                float(temp[0][0]),
                float(temp[1][0]),
                float(temp[2][0]),
                float(temp[3][0]),
                float(temp[4][0]),
                float(temp[5][0]),
                float(temp[6][0]),
                float(temp[7][0]),
                float(temp[8][0])]
            )
            x_list.append(a)
        # print(x_list)
        x_list = torch.tensor(np.array(x_list), dtype=torch.float32)
        x_list = torch.unsqueeze(x_list,2)
        # print(x_list.shape)
        # time.sleep(1)
        # print(x_list)
        # time.sleep(10)
        return x_list


class Trainmodel:
    def __init__(self,theta):
        self.batch_size = 40
        self.epochs = 500
        self.theta = theta
        np.random.seed(42)
        self.model = Nen()
        self.learning_rate = 1e-6
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.total_acc = 0
        self.total_tpr = 0
        self.total_fpr = 1

    def test(self):
        model=torch.load('test.pth')
        model.eval()
        val_data_loader = self.val_data()
        count = 0
        acc = 0
        tprn = 0
        fprn = 0
        total0 = 0
        total1 = 0
        for step, (batch_x, batch_y) in enumerate(val_data_loader):
            # print(batch_x.shape)
            pred = model(batch_x)
            # print(pred)
            out = self.dtmax(self.theta,pred)
            y = batch_y.tolist()
            # print(out)
            # print(y)
            for i in range(len(out)):
                count += 1
                if y[i] ==0:
                    total0 +=1
                else:
                    total1+=1
                if int(out[i]) == int(y[i]):
                    acc+=1

                if int(out[i]) == 1 and int(y[i])==1:
                    tprn+=1
                if int(out[i]) == 1 and int(y[i])==0:
                    fprn +=1
        # print(acc,count,tprn,fprn,total0,total1)
        # time.sleep(10)
        tpr = round((tprn/total1),3)
        fpr = round((fprn/total0),3)
        acc2 = round((acc/count),3)
        # time.sleep(100)
        return acc2 ,tpr,fpr

    def dtmax(self,theta,pred_tensor):
        relist = []
        predlist = pred_tensor.tolist()
        for i in predlist:
            if i >theta:
                relist.append(1)
            else:
                relist.append(0)
        return relist

    def train_data(self):
        y_data = []
        path = 'train.txt'
        fp = open(path, 'r')
        allfile = fp.read()
        dic = json.loads(allfile)
        x_data = np.array(dic['xdata'])
        for i in dic['ydata']:
            y_data.append(i[0])
        y_data = np.array(y_data)
        torch_dataset = Data.TensorDataset(torch.tensor(x_data), torch.tensor(y_data,dtype=torch.float32))
        loader = Data.DataLoader(
            dataset=torch_dataset,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True,  # 要不要打乱数据 (打乱比较好)
            num_workers=0,  # 多线程来读数据
        )
        return loader

    def val_data(self):
        test_y_data = []
        path = 'val.txt'
        fp = open(path, 'r')
        allfile = fp.read()
        dic = json.loads(allfile)
        # print(dic)
        test_x_data = np.array(dic['xdata'])
        for i in dic['ydata']:
            test_y_data.append(i[0])
        test_y_data = np.array(test_y_data)
        test_torch_dataset = Data.TensorDataset(torch.tensor(test_x_data), torch.tensor(test_y_data,dtype=torch.float32))
        test_loader = Data.DataLoader(
            dataset=test_torch_dataset,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True,  # 要不要打乱数据 (打乱比较好)
            num_workers=0,  # 多线程来读数据
        )
        return test_loader

    def train_epoch(self):
        for epoch in range(self.epochs):
            train_data_loader = self.train_data()
            size = len(train_data_loader)
            # print('loader_size', size)
            for step, (batch_x, batch_y) in enumerate(train_data_loader):  # 每一步 loader 释放一小批数据用来学习
                pred = self.model(batch_x)
                loss = self.loss_fn(pred, batch_y)
                loss.requires_grad_(True)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if step % 10 == 0:
                    loss, current = loss.item(), step * len(batch_x)
                    # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                    path1 = r'../nen/res/'
                    torch.save(self.model, 'test.pth')
                    acc ,tpr,fpr = self.test()
                    if acc>= self.total_acc and tpr>= self.total_tpr and fpr<=self.total_fpr:
                        path2 = str(self.theta)+'_'+str(acc)+'.pth'
                        path3 = os.path.join(path1,path2)
                        torch.save(self.model, path3)
                        self.total_acc = acc
                        self.total_tpr = tpr
                        self.total_fpr = fpr
                    # print('total acc:',self.total_acc,'roc_pos',(fpr,tpr))
        print('theta',self.theta,'max_acc',self.total_acc,'roc_pos(fpr,tpr)',(self.total_fpr,self.total_tpr))

for i in np.arange(0.6500,0.6700,0.0001):
    p = Trainmodel(i)
    p.train_epoch()



