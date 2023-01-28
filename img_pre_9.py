import argparse
import os
import shutil
import nen.net_equilibrium as ne
import cv2 as cv
import numpy as np
import math
import time
import imgkit

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer Test script', add_help=False)
    parser.add_argument('--ids')
    parser.add_argument('--imgpath')
    args, _ = parser.parse_known_args()
    return args

def bkxy(x,y):
    xpos = round(x / 512) - 1
    ypos = round(y / 512) - 1
    return xpos,ypos

def corpimg(res_img,width,height,newpath2):
    # print(3,newpath2)
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
                tiles = res_img[y:y + M, x:x + N]
                # pos transpose
                xpos,ypos = bkxy(x1,y1)
                # Save each patch into file directory
                filename = str(xpos) + '_' + str(ypos) + '.jpg'
                patha = os.path.join(newpath2,filename)
                cv.imwrite(patha, tiles)
            elif y1 >= height:  # when patch height exceeds the image height
                y1 = height - 1
                tiles = res_img[y:y + M, x:x + N]
                xpos,ypos = bkxy(x1,y1)
                filename = str(xpos) + '_' + str(ypos) + '.jpg'
                patha = os.path.join(newpath2, filename)
                cv.imwrite(patha, tiles)
            elif x1 >= width:
                x1 = width - 1
                tiles = res_img[y:y + M, x:x + N]
                xpos,ypos = bkxy(x1,y1)
                filename = str(xpos) + '_' + str(ypos) + '.jpg'
                patha = os.path.join(newpath2, filename)
                cv.imwrite(patha, tiles)
            else:
                tiles = res_img[y:y + M, x:x + N]
                xpos,ypos = bkxy(x1,y1)
                filename = str(xpos) + '_' + str(ypos) + '.jpg'
                patha = os.path.join(newpath2, filename)
                cv.imwrite(patha, tiles)

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

def predictimg(model,pth,model_acc,sp_path):
    feedback = []
    isdir = os.path.isdir(sp_path)
    # print(isdir,sp_path)
    if isdir:
        imgs = os.listdir(sp_path)
        # print(2,imgs)
        for file in imgs:
            path = os.path.join(sp_path,file)
            # print(path)
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
                    if res['Prediction'] == 'Tumor1N0_Non-Metastatic':
                        pb = round((1-res['Possibility']),4)
                    else:
                        pb = round((res['Possibility']),4)
                    feedback.append(pb)
        # print(1,feedback)
        return feedback
    else:
        path = sp_path
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
                res1 = res['Prediction']
                res2 = res['Possibility']
                res3 = str(res2*100)+'% of '+res1
                print(res3)



def run(id,input_path):
    newpath = os.path.join('dataset',id,)
    newpath2 = os.path.join(newpath,'sp9')
    isExists=os.path.exists(newpath)
    isExists2 = os.path.exists(newpath2)
    if isExists:
        pass
    else:
        os.mkdir(newpath)
    if isExists2:
        pass
    else:
        os.mkdir(newpath2)
    time1 = time.time()
    # args, _ = parse_option()
    id = id
    input_path = input_path
    img = cv.imread(input_path)
    sp = img.shape
    width = sp[1]
    height = sp[0]
    if width>1536 and height>1536:
        res_img = cv.resize(img,(1536,1536),interpolation=cv.INTER_AREA)
    elif width==1536 and height==1536:
        res_img = img
    else:
        res_img = cv.resize(img, (1536,1536), interpolation=cv.INTER_CUBIC)
    width = 1536
    height = 1536
    corpimg(res_img,width,height,newpath2)
    model1 = 'configs/swinv2/swinv2_tiny_patch4_window16_256-512.yaml '
    pth1_1 = 'predictpath/1008-ep81-rgb-97.41.pth '
    pth1_2 = 'predictpath/1021-ep141-gry-86.69.pth '
    model_acc_1_1 = 97.41
    model_acc_1_2 = 86.69
    feedback1 = predictimg(model1,pth1_1,model_acc_1_1,newpath2)
    feedback2 = predictimg(model1, pth1_2, model_acc_1_2, newpath2)
    cmd = 'python ./nen/net_equilibrium.py  --seq ' + str(feedback1).replace(' ','') + ' --mode ' + 'RGB' + ' --theta ' + str(0.6570)
    cmd2 = 'python ./nen/net_equilibrium.py  --seq ' + str(feedback2).replace(' ','') + ' --mode ' + 'Gray' + ' --theta ' + str(0.6581)
    result = os.popen(cmd)
    res = result.read()
    for line in res.splitlines():
        if 'Prediction' in line:
            res = eval(line)
            re = res['Prediction']
            if re =="Tumor1N0_Non-Metastatic":
                fpb = 0
            else:
                fpb=1
    result2 = os.popen(cmd2)
    res2 = result2.read()
    for line in res2.splitlines():
        if 'Prediction' in line:
            res2 = eval(line)
            re2 = res2['Prediction']
            if re2 =="Tumor1N0_Non-Metastatic":
                fpb2 = 0
            else:
                fpb2=1
    final_pred = (fpb*model_acc_1_1*0.5)+ (fpb2*model_acc_1_2*0.6)/( model_acc_1_1+model_acc_1_2)
    if final_pred>0.5:
        print('Tumor1N1_Metastatic')
    else:
        print('Tumor1N0_Non-Metastatic')



    time2 = time.time()
    delta = round(time2-time1,2)
    print('总耗时',delta,'秒')

if __name__ == '__main__':
    args = parse_option()
    id = args.ids
    input_path = args.imgpath
    run(id,input_path)
