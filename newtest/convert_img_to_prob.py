import json
import math
import os
import time
import cv2 as cv
import numpy as np


class FITV():
    def __init__(self):
        self.train_n0_path = '../newtest/nendataset/train/n0'
        self.train_n1_path = '../newtest/nendataset/train/n1'
        self.val_n0_path = '../newtest/nendataset/val/n0'
        self.val_n1_path = '../newtest/nendataset/val/n1'
        self.sp_path = '../newtest/nendataset/sp9'
        self.model1 = '../configs/swinv2/swinv2_tiny_patch4_window16_256-512.yaml '
        self.pth1_1 = '../pridictpath/1008-ep81-rgb-97.41.pth '
        self.pth1_2 = '../pridictpath/1021-ep141-gry-86.69.pth '
        self.model_acc_1_1 = 97.41
        self.model_acc_1_2 = 86.69
        self.traindata = '../newtest/train.txt'
        self.valdata = '../newtest/val.txt'


    def crop_img_9(self,img_path):
        img = cv.imread(img_path)
        if img.shape[1]>1536 and img.shape[0]>1536:
            res_img = cv.resize(img,(1536,1536),interpolation=cv.INTER_AREA)
        elif img.shape[1]==1536 and img.shape[0]==1536:
            res_img = img
        else:
            res_img = cv.resize(img, (1536,1536), interpolation=cv.INTER_CUBIC)
        self.corpimg(res_img,1536,1536,self.sp_path)

    def corpimg(self,res_img, width, height, newpath2):
        M = 512
        N = 512
        def bkxy(x, y):
            xpos = round(x / 512) - 1
            ypos = round(y / 512) - 1
            return xpos, ypos
        for y in range(0, height, M):
            for x in range(0, width, N):
                if (height - y) < M or (width - x) < N:
                    break
                y1 = y + M
                x1 = x + N
                # check whether the patch width or height exceeds the image width or height
                if x1 >= width and y1 >= height:
                    x1 = width - 1
                    y1 = height - 1
                    # Crop into patches of size MxN
                    tiles = res_img[y:y + M, x:x + N]
                    # pos transpose
                    xpos, ypos = bkxy(x1, y1)
                    # Save each patch into file directory
                    filename = str(xpos) + '_' + str(ypos) + '.jpg'
                    patha = os.path.join(newpath2, filename)
                    cv.imwrite(patha, tiles)
                elif y1 >= height:  # when patch height exceeds the image height
                    y1 = height - 1
                    tiles = res_img[y:y + M, x:x + N]
                    xpos, ypos = bkxy(x1, y1)
                    filename = str(xpos) + '_' + str(ypos) + '.jpg'
                    patha = os.path.join(newpath2, filename)
                    cv.imwrite(patha, tiles)
                elif x1 >= width:
                    x1 = width - 1
                    tiles = res_img[y:y + M, x:x + N]
                    xpos, ypos = bkxy(x1, y1)
                    filename = str(xpos) + '_' + str(ypos) + '.jpg'
                    patha = os.path.join(newpath2, filename)
                    cv.imwrite(patha, tiles)
                else:
                    tiles = res_img[y:y + M, x:x + N]
                    xpos, ypos = bkxy(x1, y1)
                    filename = str(xpos) + '_' + str(ypos) + '.jpg'
                    patha = os.path.join(newpath2, filename)
                    cv.imwrite(patha, tiles)

    def get_conf(self,path):
        img = cv.imread(path)
        img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img_canny = cv.Canny(img_gray, 0, 100)
        kernel = np.ones((5, 5), np.uint8)
        dilate = cv.dilate(img_canny, kernel)
        dilate2 = cv.dilate(dilate, kernel)
        dilate3 = cv.dilate(dilate2, kernel)
        h, w = img_gray.shape
        dst = np.zeros((h, w, 1), np.uint8)
        for i in range(h):
            for j in range(w):
                dst[i, j] = 255 - dilate3[i, j]
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(dst, connectivity=8)
        output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for i in range(1, num_labels):
            mask = labels == i
            output[:, :, 0][mask] = np.random.randint(0, 255)
            output[:, :, 1][mask] = np.random.randint(0, 255)
            output[:, :, 2][mask] = np.random.randint(0, 255)
        s_list = []
        s_dict = {}
        for row in range(h):  # 对图中所有的像素点进行遍历
            for col in range(w):
                s = dst[row, col]
                if s in s_list:
                    temp2 = s_dict['%s' % s]
                    s_dict['%s' % s] = temp2 + 1
                else:
                    s_list.append(s)
                    s_dict['%s' % s] = 1
        # print(s_list)
        # print(s_dict)
        total = h * w
        max_area = s_dict['[255]']
        # print(max_area)
        percent = round(max_area / total, 3)
        # print(percent)
        if percent <= 0.1:
            prob = 1
        else:
            prob = round(-math.log(percent, 10), 3)  # 使用math中的log函数生成对应x的值
        # print(prob)
        return percent, prob

    def pred_img(self,model,pth,model_acc,path):
        pc, pb = self.get_conf(path)
        # print(file,pc,pb)
        cmd = 'python ../pd.py  --cfg ' + model + '--ckp_path ' + pth + '--img_path ' + str(path) + ' --model_acc ' + str(
            model_acc)
        result = os.popen(cmd)
        res = result.read()
        for line in res.splitlines():
            # print(line)
            if 'Prediction' in line:
                # print(type(line))
                res = eval(line)
                res['Confidence'] = pb
        return  res

    def conv_res_to_scalar(self,res):
        pb = round(res['Possibility'] * (float(res['Model_Acc'])/100),4)
        if res['Prediction'] == 'Tumor1N0_Non-Metastatic':
            pb = (1-pb)
        return pb

    def create_tran_xy(self):
        imgs = os.listdir(self.train_n0_path)
        xdata=[]
        ydata = []
        for file in imgs:
            input_path = os.path.join(self.train_n0_path,file)
            self.crop_img_9(input_path)
            sps = os.listdir(self.sp_path)
            # print(2,imgs)
            trainx = []
            for file in sps:
                path = os.path.join(self.sp_path, file)
                res = self.pred_img(self.model1,self.pth1_1,self.model_acc_1_1,path)
                # print(res)
                pb = self.conv_res_to_scalar(res)
                trainx.append(pb)
            xdata.append(trainx)
            ydata.append([0])
        imgs2 = os.listdir(self.train_n1_path)
        for file in imgs2:
            input_path = os.path.join(self.train_n1_path,file)
            self.crop_img_9(input_path)
            sps = os.listdir(self.sp_path)
            # print(2,imgs)
            trainx = []
            for file in sps:
                path = os.path.join(self.sp_path, file)
                res = self.pred_img(self.model1,self.pth1_1,self.model_acc_1_1,path)
                # print(res)
                pb = self.conv_res_to_scalar(res)
                trainx.append(pb)
            xdata.append(trainx)
            ydata.append([1])
        d = {}
        d['xdata'] = xdata
        d['ydata'] = ydata
        djson = json.dumps(d,sort_keys=False,indent=4,separators=(',',':'))
        f = open(self.traindata,'w')
        f.write(djson)

    def create_val_xy(self):
        imgs = os.listdir(self.val_n0_path)
        xdata=[]
        ydata = []
        for file in imgs:
            input_path = os.path.join(self.val_n0_path,file)
            self.crop_img_9(input_path)
            sps = os.listdir(self.sp_path)
            # print(2,imgs)
            valx = []
            for file in sps:
                path = os.path.join(self.sp_path, file)
                res = self.pred_img(self.model1,self.pth1_1,self.model_acc_1_1,path)
                # print(res)
                pb = self.conv_res_to_scalar(res)
                valx.append(pb)
            xdata.append(valx)
            ydata.append([0])
        imgs2 = os.listdir(self.val_n1_path)
        for file in imgs2:
            input_path = os.path.join(self.val_n1_path,file)
            self.crop_img_9(input_path)
            sps = os.listdir(self.sp_path)
            # print(2,imgs)
            valx = []
            for file in sps:
                path = os.path.join(self.sp_path, file)
                res = self.pred_img(self.model1,self.pth1_1,self.model_acc_1_1,path)
                # print(res)
                pb = self.conv_res_to_scalar(res)
                valx.append(pb)
            xdata.append(valx)
            ydata.append([1])
        d = {}
        d['xdata'] = xdata
        d['ydata'] = ydata
        djson = json.dumps(d,sort_keys=False,indent=4,separators=(',',':'))
        f = open(self.valdata,'w')
        f.write(djson)


p = FITV()
# p.create_tran_xy()
p.create_val_xy()