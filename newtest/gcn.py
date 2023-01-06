import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable



class dcn(nn.Module):  # 定义网络
    def __init__(self):
        super(dcn, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.theta = theta
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

def test():
    model=torch.load('test.pth')
    model.eval()
    val_data_loader = val_data()
    for step, (batch_x, batch_y) in enumerate(val_data_loader):  
        # print(batch_x.shape)
        pred = model(batch_x)
        out = dtmax(theta,pred)
        # print(out)
        # print(batch_y)
        count = 0
        acc = 0
        for i in range(len(out)):

            if int(out[i]) == int(batch_y.tolist()[i]):
                count+=1
                acc+=1
            else:
                count+=1
        acc2 = round((acc/count),3)
        # time.sleep(100)
        return acc2

def dtmax(theta,pred_tensor):
    relist = []
    predlist = pred_tensor.tolist()
    for i in predlist:
        if i >theta:
            relist.append(1)
        else:
            relist.append(0)
    return relist

def train_data():
    y_data = []
    # for i in range(8192):
    #     x = np.random.uniform(0, 1, (9, 1))
    #     x_data.append(x)
    # x_data = np.array(x_data)
    # y_data = np.random.randint(0, 2, (8192,))
    # y_data = np.array([[i] for i in y_data])
    path = 'train.txt'
    fp = open(path, 'r')
    allfile = fp.read()
    dic = json.loads(allfile)
    x_data = np.array(dic['xdata'])
    for i in dic['ydata']:
        y_data.append(i[0])
    y_data = np.array(y_data)
    # print(len(dic['xdata']))
    # print(len(dic['ydata']))
    torch_dataset = Data.TensorDataset(torch.tensor(x_data), torch.tensor(y_data,dtype=torch.float32))
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    return loader

def val_data():
    test_y_data = []
    # for i in range(2048):
    #     x = np.random.uniform(0, 1, (9, 1))
    #     test_x_data.append(x)
    # test_x_data = np.array(test_x_data)
    # test_y_data = np.random.randint(0, 2, (2048,))
    # y_data = np.array([[i] for i in y_data])
    path = 'val.txt'
    fp = open(path, 'r')
    allfile = fp.read()
    dic = json.loads(allfile)
    # print(dic)
    test_x_data = np.array(dic['xdata'])
    for i in dic['ydata']:
        test_y_data.append(i[0])
    test_y_data = np.array(test_y_data)
    # print(len(dic['xdata']))
    # print(len(dic['ydata']))

    test_torch_dataset = Data.TensorDataset(torch.tensor(test_x_data), torch.tensor(test_y_data,dtype=torch.float32))
    test_loader = Data.DataLoader(
        dataset=test_torch_dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    return test_loader


batch_size = 40
epochs = 50000
theta = 0.5
np.random.seed(42)
model = dcn()
learning_rate = 1e-7
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
total_acc = 0

for epoch in range(epochs):
    train_data_loader = train_data()
    size = len(train_data_loader)
    # print('loader_size', size)
    for step, (batch_x, batch_y) in enumerate(train_data_loader):  # 每一步 loader 释放一小批数据用来学习
        # print(batch_x)
        # print(batch_y)
        # time.sleep(100)
        pred = model(batch_x)
        # print(pred)
        # print(batch_y)
        # print(pred)
        # # time.sleep(5)
        # print('--------------')
        loss = loss_fn(pred, batch_y)
        # print(pred.shape,batch_y.shape)
        # print(loss)
        # time.sleep(10)
        loss.requires_grad_(True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            loss, current = loss.item(), step * len(batch_x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            torch.save(model, 'test.pth')
            acc = test()
            if acc> total_acc:
                paths = str(acc)+'.pth'
                torch.save(model, paths)
                total_acc = acc
            print('total acc:',total_acc)
            # time.sleep(100)
            # writer.add_scalar('training loss',loss / 100, t*len(x)+batch//100)

        # print('Epoch: ', epoch, '| Step:', step, '| batch x: ', batch_x.numpy(), '| batch y: ', batch_y.numpy())



