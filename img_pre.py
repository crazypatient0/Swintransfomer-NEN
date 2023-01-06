import os
import cv2 as cv
import numpy as np
import math


from matplotlib import pyplot as plt

input_path = "dataset/11/Tumer1N0.16479.jpg"
img = cv.imread(input_path)#读取图片
sp = img.shape
image_copy = img.copy()
width = sp[1]
height = sp[0]

#然后判断图片是否大于1024*1024，是就缩小 否就放大
if width>1024 and height>1024:
    res = cv.resize(img,(1024,1024),interpolation=cv.INTER_AREA)
elif width==1024 and height==1024:
    res = img
else:
    res = cv.resize(img, (1024,1024), interpolation=cv.INTER_CUBIC)

# cv.imshow("res",res)
# cv.waitKey(0)



# 将xy的像素位置 转为01
def bkxy(x,y):
    if x == 512:
        xpos = 0
    else:
        xpos = 1
    if y == 512:
        ypos = 0
    else:
        ypos = 1
    return xpos,ypos

def corpimg(img,width,height):
    M = 512
    N = 512
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
                tiles = image_copy[y:y + M, x:x + N]
                # pos transpose
                xpos,ypos = bkxy(x1,y1)
                # Save each patch into file directory
                cv.imwrite('dataset/sp/' + str(xpos) + '_' + str(ypos) + '.jpg', tiles)
                cv.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
            elif y1 >= height:  # when patch height exceeds the image height
                y1 = height - 1
                # Crop into patches of size MxN
                tiles = image_copy[y:y + M, x:x + N]
                # pos transpose
                xpos,ypos = bkxy(x1,y1)
                # Save each patch into file directory
                cv.imwrite('dataset/sp/' + str(xpos) + '_' + str(ypos) + '.jpg', tiles)
                cv.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
            elif x1 >= width:  # when patch width exceeds the image width
                x1 = width - 1
                # Crop into patches of size MxN
                tiles = image_copy[y:y + M, x:x + N]
                # pos transpose
                xpos,ypos = bkxy(x1,y1)
                # Save each patch into file directory
                cv.imwrite('dataset/sp/' + str(xpos) + '_' + str(ypos) + '.jpg', tiles)
                cv.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
            else:
                # Crop into patches of size MxN
                tiles = image_copy[y:y + M, x:x + N]
                # pos transpose
                xpos,ypos = bkxy(x1,y1)
                # Save each patch into file directory
                cv.imwrite('dataset/sp/' + str(xpos) + '_' + str(ypos) + '.jpg', tiles)
                cv.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)

def get_conf(path):
    img = cv.imread(path)
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img_canny = cv.Canny(img_gray,0,100)
    kernel = np.ones((5,5), np.uint8)
    dilate = cv.dilate(img_canny, kernel)
    dilate2 = cv.dilate(dilate, kernel)
    dilate3 = cv.dilate(dilate2, kernel)
    h, w = img_gray.shape
    dst=np.zeros((h,w,1),np.uint8)
    for i in range(h):
        for j in range(w):
            dst[i,j]=255-dilate3[i,j]
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
                temp2 = s_dict['%s'%s]
                s_dict['%s' % s] = temp2 +1
            else:
                s_list.append(s)
                s_dict['%s'%s] = 1
    # print(s_list)
    # print(s_dict)
    total = h*w
    max_area = s_dict['[255]']
    # print(max_area)
    percent = round(max_area/total,3)
    # print(percent)
    if percent<=0.1:
        prob = 1
    else:
        prob = round(-math.log(percent,10),3) # 使用math中的log函数生成对应x的值
    # print(prob)
    return percent,prob

corpimg(img,width,height)

def predictimg(model,pth,model_acc):
    feedback = []
    sp_path = 'dataset/sp/'
    imgs = os.listdir(sp_path)
    for file in imgs:
        path = os.path.join(sp_path,file)
        #pc- empty_area_percent,pb- area_confidence_probability
        pc,pb = get_conf(path)
        # print(file,pc,pb)
        cmd = 'python pd.py  --cfg '+model+'--ckp_path '+pth+'--img_path '+str(path)+' --model_acc '+str(model_acc)
        result = os.popen(cmd)
        res = result.read()
        for line in res.splitlines():
            # print(line)
            if 'Prediction' in line:
                # print(type(line))
                res = eval(line)
                res['img'] = file
                res['Confidence'] = pb
                res['Empty_area'] = str(round(pc*100,2))+'%'
                feedback.append(res)
    return feedback

model1 = 'configs/swinv2/swinv2_tiny_patch4_window16_256-512.yaml '
pth1_1 = 'pridictpath/1008-ep81-rgb-97.41.pth '
pth1_2 = 'pridictpath/1021-ep141-gry-86.69.pth '
model_acc_1_1 = 97.41
model_acc_1_2 = 86.69
feedback2 = predictimg(model1,pth1_2,model_acc_1_2)
print(feedback2)
def iterationimg(feedback):
    for i in feedback:
        if i['img'] == '0_0.jpg':
            if i['Prediction'] == 'Tumor1N0_Non-Metastatic':
                a00_pb = round(100-float(i['Possibility']),3)
            else:
                a00_pb = i['Possibility']
        if i['img'] == '0_1.jpg':
            if i['Prediction'] == 'Tumor1N0_Non-Metastatic':
                a01_pb = round(100-float(i['Possibility']),3)
            else:
                a01_pb = i['Possibility']
        if i['img'] == '1_0.jpg':
            if i['Prediction'] == 'Tumor1N0_Non-Metastatic':
                a10_pb = round(100-float(i['Possibility']),3)
            else:
                a10_pb = i['Possibility']
        if i['img'] == '1_1.jpg':
            if i['Prediction'] == 'Tumor1N0_Non-Metastatic':
                a11_pb = round(100-float(i['Possibility']),3)
            else:
                a11_pb = i['Possibility']
    print(a00_pb,a01_pb,a10_pb,a11_pb)
    while 1:
        tmp00 = round((0.7*a00_pb+0.3*(a01_pb+a10_pb)/2),3)
        tmp01 = round((0.7 * a01_pb + 0.3 * (a00_pb + a11_pb) / 2), 3)
        tmp10 = round((0.7 * a10_pb + 0.3 * (a00_pb + a11_pb) / 2), 3)
        tmp11 = round((0.7 * a11_pb + 0.3 * (a01_pb + a10_pb) / 2), 3)
        if abs(tmp00 - a00_pb)>0.001 or abs(tmp01 - a01_pb)>0.001 or abs(tmp10 - a10_pb)>0.001 or abs(tmp11 - a11_pb)>0.001:
            a00_pb = tmp00
            a01_pb = tmp01
            a10_pb = tmp10
            a11_pb = tmp11
        else:
            break
    print(a00_pb, a01_pb, a10_pb, a11_pb)

iterationimg(feedback2)



# predictimg(model1,pth1_1,conf1_1)