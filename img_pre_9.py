import argparse
import os
import shutil

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
                    res['img'] = file
                    res['Confidence'] = pb
                    res['Empty_area'] = str(round(pc*100,2))+'%'
                    feedback.append(res)
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

def iterationimg(feedback):
    # print(feedback)
    for i in feedback:
        if i['img'] == '0_0.jpg':
            if i['Prediction'] == 'Tumor1N0_Non-Metastatic':
                a00_pb = (1-float(i['Possibility'])) * float(i['Confidence'])
            else:
                a00_pb = i['Possibility']  * float(i['Confidence'])
        if i['img'] == '0_1.jpg':
            if i['Prediction'] == 'Tumor1N0_Non-Metastatic':
                a01_pb = (1-float(i['Possibility']))  * float(i['Confidence'])
            else:
                a01_pb = i['Possibility']  * float(i['Confidence'])
        if i['img'] == '0_2.jpg':
            if i['Prediction'] == 'Tumor1N0_Non-Metastatic':
                a02_pb = (1-float(i['Possibility']))  * float(i['Confidence'])
            else:
                a02_pb = i['Possibility']  * float(i['Confidence'])
        if i['img'] == '1_0.jpg':
            if i['Prediction'] == 'Tumor1N0_Non-Metastatic':
                a10_pb = (1-float(i['Possibility']))  * float(i['Confidence'])
            else:
                a10_pb = i['Possibility']  * float(i['Confidence'])
        if i['img'] == '1_1.jpg':
            if i['Prediction'] == 'Tumor1N0_Non-Metastatic':
                a11_pb = (1-float(i['Possibility']))  * float(i['Confidence'])
            else:
                a11_pb = i['Possibility']  * float(i['Confidence'])
        if i['img'] == '1_2.jpg':
            if i['Prediction'] == 'Tumor1N0_Non-Metastatic':
                a12_pb = (1-float(i['Possibility']))  * float(i['Confidence'])
            else:
                a12_pb = i['Possibility']  * float(i['Confidence'])
        if i['img'] == '2_0.jpg':
            if i['Prediction'] == 'Tumor1N0_Non-Metastatic':
                a20_pb = (1-float(i['Possibility']))  * float(i['Confidence'])
            else:
                a20_pb = i['Possibility']  * float(i['Confidence'])
        if i['img'] == '2_1.jpg':
            if i['Prediction'] == 'Tumor1N0_Non-Metastatic':
                a21_pb = (1-float(i['Possibility']))  * float(i['Confidence'])
            else:
                a21_pb = i['Possibility']  * float(i['Confidence'])
        if i['img'] == '2_2.jpg':
            if i['Prediction'] == 'Tumor1N0_Non-Metastatic':
                a22_pb = (1-float(i['Possibility']))  * float(i['Confidence'])
            else:
                a22_pb = i['Possibility']  * float(i['Confidence'])
    # print(a00_pb,a01_pb,a02_pb,a10_pb,a11_pb,a12_pb,a20_pb,a21_pb,a22_pb)
    while 1:
        tmp00 = round((0.5 * a00_pb + 0.5 * (a01_pb + a10_pb + a11_pb) / 3),3)
        tmp01 = round((0.5 * a01_pb + 0.5 * (a00_pb + a10_pb + a11_pb + a12_pb + a02_pb) / 5), 3)
        tmp02 = round((0.5 * a02_pb + 0.5 * (a01_pb + a11_pb + a12_pb) / 3), 3)
        tmp10 = round((0.5 * a10_pb + 0.5 * (a00_pb + a01_pb + a11_pb + a21_pb + a20_pb) / 5), 3)
        tmp11 = round((0.5 * a11_pb + 0.5 * (a00_pb + a01_pb + a02_pb + a10_pb + a12_pb + a20_pb + a21_pb + a22_pb) / 8), 3)
        tmp12 = round((0.5 * a12_pb + 0.5 * (a02_pb + a01_pb + a11_pb + a21_pb + a22_pb) / 5), 3)
        tmp20 = round((0.5 * a20_pb + 0.5 * (a21_pb + a11_pb + a10_pb) / 3),3)
        tmp21 = round((0.5 * a21_pb + 0.5 * (a20_pb + a10_pb + a11_pb + a12_pb + a22_pb) / 5), 3)
        tmp22 = round((0.5 * a22_pb + 0.5 * (a21_pb + a11_pb + a12_pb) / 3), 3)
        if abs(tmp00 - a00_pb)>0.001 or abs(tmp01 - a01_pb)>0.001 or abs(tmp02 - a02_pb)>0.001  or abs(tmp10 - a10_pb)>0.001 or abs(tmp11 - a11_pb)>0.001 or abs(tmp12 - a12_pb)>0.001 or abs(tmp20 - a20_pb)>0.001 or abs(tmp21 - a21_pb)>0.001 or abs(tmp22 - a22_pb)>0.001:
            a00_pb = tmp00
            a01_pb = tmp01
            a02_pb = tmp02
            a10_pb = tmp10
            a11_pb = tmp11
            a12_pb = tmp12
            a20_pb = tmp20
            a21_pb = tmp21
            a22_pb = tmp22
        else:
            break
    # print(a00_pb)
    return a00_pb

def run(id,input_path):
    newpath = os.path.join('dataset',id,)
    newpath2 = os.path.join(newpath,'sp9')
    newpath3 = os.path.join('dataset', id,'input.jpg' )
    newpath4 = os.path.join('dataset', id, 'gcn.gif')
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
    shutil.copyfile(input_path, newpath3)
    shutil.copyfile('gcn.gif', newpath4)
    # print(newpath)
    # print(newpath2)
    time1 = time.time()
    # args, _ = parse_option()
    id = id
    input_path = input_path
    img = cv.imread(input_path)#读取图片
    sp = img.shape
    image_copy = img.copy()
    width = sp[1]
    height = sp[0]
    #然后判断图片是否大于1536*1536，是就缩小 否就放大
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
    pth1_1 = 'pridictpath/1008-ep81-rgb-97.41.pth '
    pth1_2 = 'pridictpath/1021-ep141-gry-86.69.pth '
    model_acc_1_1 = 97.41
    model_acc_1_2 = 86.69
    feedback3 = predictimg(model1,pth1_1,model_acc_1_1,input_path)
    feedback2 = predictimg(model1,pth1_2,model_acc_1_2,newpath2)
    feedback1 = predictimg(model1,pth1_1,model_acc_1_1,newpath2)
    # for i in feedback1:
    #     print(i)
    # print(feedback1)
    # print('---------------')
    # print(feedback2)
    avg_prob2 = iterationimg(feedback2)
    avg_prob1 = iterationimg(feedback1)
    total_prob = (model_acc_1_2*avg_prob2)/(model_acc_1_2+model_acc_1_1)+(model_acc_1_1*avg_prob1)/(model_acc_1_2+model_acc_1_1)
    # print("total_prob",total_prob)
    # 画热力图
    heat_map2 = 'python heat_map.py --cfg '+model1 +'--data-path '+input_path+' --pretrained '+pth1_2+' --local_rank 0 --imgname '+newpath+'/heat_map2.jpg'
    os.system(heat_map2)

    heat_map1 = 'python heat_map.py --cfg '+model1 +'--data-path '+input_path+' --pretrained '+pth1_1+' --local_rank 0 --imgname '+newpath+'/heat_map1.jpg'
    os.system(heat_map1)

    #设置网页变量
    s1 = id
    s2 = 'input.jpg'
    if feedback1[0]['Prediction'] == 'Tumor1N0_Non-Metastatic':
        s3 = str(round((1-float(feedback1[0]['Possibility']))  * float(feedback1[0]['Confidence']) ,4))
    else:
        s3 = str(round(float(feedback1[0]['Possibility']) * float(feedback1[0]['Confidence']), 4))
    if feedback1[1]['Prediction'] == 'Tumor1N0_Non-Metastatic':
        s4 = str(round((1-float(feedback1[1]['Possibility']))  * float(feedback1[1]['Confidence']) ,4))
    else:
        s4 = str(round(float(feedback1[1]['Possibility']) * float(feedback1[1]['Confidence']), 4))
    if feedback1[2]['Prediction'] == 'Tumor1N0_Non-Metastatic':
        s5 = str(round((1-float(feedback1[2]['Possibility']))  * float(feedback1[2]['Confidence']) ,4))
    else:
        s5 = str(round(float(feedback1[2]['Possibility']) * float(feedback1[2]['Confidence']), 4))
    if feedback1[3]['Prediction'] == 'Tumor1N0_Non-Metastatic':
        s6 = str(round((1-float(feedback1[3]['Possibility']))  * float(feedback1[3]['Confidence']) ,4))
    else:
        s6 = str(round(float(feedback1[3]['Possibility']) * float(feedback1[3]['Confidence']), 4))
    if feedback1[4]['Prediction'] == 'Tumor1N0_Non-Metastatic':
        s7 = str(round((1-float(feedback1[4]['Possibility']))  * float(feedback1[4]['Confidence']) ,4))
    else:
        s7 = str(round(float(feedback1[4]['Possibility']) * float(feedback1[4]['Confidence']), 4))
    if feedback1[5]['Prediction'] == 'Tumor1N0_Non-Metastatic':
        s8 = str(round((1-float(feedback1[5]['Possibility']))  * float(feedback1[5]['Confidence']) ,4))
    else:
        s8 = str(round(float(feedback1[5]['Possibility']) * float(feedback1[5]['Confidence']), 4))
    if feedback1[6]['Prediction'] == 'Tumor1N0_Non-Metastatic':
        s9 = str(round((1-float(feedback1[6]['Possibility']))  * float(feedback1[6]['Confidence']) ,4))
    else:
        s9 = str(round(float(feedback1[6]['Possibility']) * float(feedback1[6]['Confidence']), 4))
    if feedback1[7]['Prediction'] == 'Tumor1N0_Non-Metastatic':
        s10 = str(round((1-float(feedback1[7]['Possibility']))  * float(feedback1[7]['Confidence']) ,4))
    else:
        s10 = str(round(float(feedback1[7]['Possibility']) * float(feedback1[7]['Confidence']), 4))
    if feedback1[8]['Prediction'] == 'Tumor1N0_Non-Metastatic':
        s11 = str(round((1-float(feedback1[8]['Possibility']))  * float(feedback1[8]['Confidence']) ,4))
    else:
        s11 = str(round(float(feedback1[8]['Possibility']) * float(feedback1[8]['Confidence']), 4))
    if feedback2[0]['Prediction'] == 'Tumor1N0_Non-Metastatic':
        s12 = str(round((1-float(feedback2[0]['Possibility']))  * float(feedback2[0]['Confidence']) ,4))
    else:
        s12 = str(round(float(feedback2[0]['Possibility']) * float(feedback2[0]['Confidence']), 4))
    if feedback2[1]['Prediction'] == 'Tumor1N0_Non-Metastatic':
        s13 = str(round((1-float(feedback2[1]['Possibility']))  * float(feedback2[1]['Confidence']) ,4))
    else:
        s13 = str(round(float(feedback2[1]['Possibility']) * float(feedback2[1]['Confidence']), 4))
    if feedback2[2]['Prediction'] == 'Tumor1N0_Non-Metastatic':
        s14 = str(round((1-float(feedback2[2]['Possibility']))  * float(feedback2[2]['Confidence']) ,4))
    else:
        s14 = str(round(float(feedback2[2]['Possibility']) * float(feedback2[2]['Confidence']), 4))
    if feedback2[3]['Prediction'] == 'Tumor1N0_Non-Metastatic':
        s15 = str(round((1-float(feedback2[3]['Possibility']))  * float(feedback2[3]['Confidence']) ,4))
    else:
        s15 = str(round(float(feedback2[3]['Possibility']) * float(feedback2[3]['Confidence']), 4))
    if feedback2[4]['Prediction'] == 'Tumor1N0_Non-Metastatic':
        s16 = str(round((1-float(feedback2[4]['Possibility']))  * float(feedback2[4]['Confidence']) ,4))
    else:
        s16 = str(round(float(feedback2[4]['Possibility']) * float(feedback2[4]['Confidence']), 4))
    if feedback2[5]['Prediction'] == 'Tumor1N0_Non-Metastatic':
        s17 = str(round((1-float(feedback2[5]['Possibility']))  * float(feedback2[5]['Confidence']) ,4))
    else:
        s17 = str(round(float(feedback2[5]['Possibility']) * float(feedback2[5]['Confidence']), 4))
    if feedback2[6]['Prediction'] == 'Tumor1N0_Non-Metastatic':
        s18 = str(round((1-float(feedback2[6]['Possibility']))  * float(feedback2[6]['Confidence']) ,4))
    else:
        s18 = str(round(float(feedback2[6]['Possibility']) * float(feedback2[6]['Confidence']), 4))
    if feedback2[7]['Prediction'] == 'Tumor1N0_Non-Metastatic':
        s19 = str(round((1-float(feedback2[7]['Possibility']))  * float(feedback2[7]['Confidence']) ,4))
    else:
        s19 = str(round(float(feedback2[7]['Possibility']) * float(feedback2[7]['Confidence']), 4))
    if feedback2[8]['Prediction'] == 'Tumor1N0_Non-Metastatic':
        s20 = str(round((1-float(feedback2[8]['Possibility']))  * float(feedback2[8]['Confidence']) ,4))
    else:
        s20 = str(round(float(feedback2[8]['Possibility']) * float(feedback2[8]['Confidence']), 4))


    s21 = str(round(avg_prob1,3))
    s22 = str(round(avg_prob2,3))
    treshold = 0.6
    if total_prob>= treshold:
        s23 = str(round(total_prob*100 ,3)) +'% of Metastatic'
    else:
        s23 = 'Non-Metastatic'
    print(s23)
    s24 = ''
    def createweb(s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,path):
        filename = 'report.html'
        path = path
        pathr = os.path.join(path,filename)
        imgpath1 =os.path.join('sp9','0_0.jpg')
        imgpath2 =os.path.join('sp9','0_1.jpg')
        imgpath3 =os.path.join('sp9','0_2.jpg')
        imgpath4 =os.path.join('sp9','1_0.jpg')
        imgpath5 =os.path.join('sp9','1_1.jpg')
        imgpath6 =os.path.join('sp9','1_2.jpg')
        imgpath7 =os.path.join('sp9','2_0.jpg')
        imgpath8 =os.path.join('sp9','2_1.jpg')
        imgpath9 =os.path.join('sp9','2_2.jpg')
        heat_map1 = 'heat_map1.jpg'
        heat_map2 = 'heat_map2.jpg'
        f = open(pathr,'w')
        message = """
        <html>
        <head></head>
        <body>
        <div class="main">
        <div class="head">
        <p class="head_content">Prediction Reports</p>
        </div>
        <div >
        <p >"""+s1+"""</p>
        </div>
        <div class="label">
        <div class="labelbox">
        <p>
        INPUT_resize(1536*1536)
        </p>
        </div>
        <div class="labelbox">
        <p>
        PATCH_3*3(512*512)
        </p>
        </div>
        </div>
        <div class="imgcontainer">
        <div class="input">
        <img src=\""""+s2+"""\" width="300px" height="300px" margin-top="40px"/>
        </div>
        <div class="crop">
        <div class="block9">
        <div class="patchs">
        <div class="patch">
        <img src=\""""+imgpath1+"""\" width="100px" height="100px"/>
        <span>Block 0_0</span>
        </div>
        <div class="patch">
        <img src=\""""+imgpath2+"""\" width="100px" height="100px"/>
        <span>Block 0_1</span>
        </div>
        <div class="patch">
        <img src=\""""+imgpath3+"""\" width="100px" height="100px"/>
        <span>Block 0_2</span>
        </div>
        </div>
        <div class="patchs">
        <div class="patch">
        <img src=\""""+imgpath4+"""\" width="100px" height="100px"/>
        <span>Block 1_0</span>
        </div>
        <div class="patch">
        <img src=\""""+imgpath5+"""\" width="100px" height="100px"/>
        <span>Block 1_1</span>
        </div>
        <div class="patch">
        <img src=\""""+imgpath6+"""\" width="100px" height="100px"/>
        <span>Block 1_2</span>
        </div>
        </div>
        <div class="patchs">
        <div class="patch">
        <img src=\""""+imgpath7+"""\" width="100px" height="100px"/>
        <span>Block 2_0</span>
        </div>
        <div class="patch">
        <img src=\""""+imgpath8+"""\" width="100px" height="100px"/>
        <span>Block 2_1</span>
        </div>
        <div class="patch">
        <img src=\""""+imgpath9+"""\" width="100px" height="100px"/>
        <span>Block 2_2</span>
        </div>
        </div>
        </div>
        </div>
        </div>
        <div class="title">
        <span>Global Confidence Probability</span>
        </div>
        <div class="text">
        <span class="discribe">
        <br/>
        During the image preprocessing process, a global <br/>
        confidence probability will be assigned to each patch<br/>
        according to the percentage of the cavity area<br/><br/>
        </span>
        </div>
        <div class="title">
        <span>Model Prediction - Swin Transformer (RBG PATTERN)</span>
        </div>
        <div class="text">
        <span class="discribe">
        <br/>
        Possibility - means the chance of this block to be Metastatic-Tumor<br/>
        Possibility = Confidence * Model_acc * Prediction_Possibility<br/>
        <br/>
        </span>
        <span>Possibility of Block 0_0 = """+s3+"""</span>
        <span>Possibility of Block 0_1 = """+s4+"""</span>
        <span>Possibility of Block 0_2 = """+s5+"""</span>
        <span>Possibility of Block 1_0 = """+s6+"""</span>
        <span>Possibility of Block 1_1 = """+s7+"""</span>
        <span>Possibility of Block 1_2 = """+s8+"""</span>
        <span>Possibility of Block 2_0 = """+s9+"""</span>
        <span>Possibility of Block 2_1 = """+s10+"""</span>
        <span>Possibility of Block 2_2 = """+s11+"""</span>
        <br/>
        </div>
        <div class="title">
        <span>Model Prediction - Swin Transformer (BINARY PATTERN)</span>
        </div>
        <div class="text">
        <span class="discribe">
        <br/>
        Possibility - means the chance of this block to be Metastatic-Tumor<br/>
        Possibility = Confidence * Model_acc * Prediction_Possibility<br/>
        <br/>
        </span>
        <span>Possibility of Block 0_0 = """+s12+"""</span>
        <span>Possibility of Block 0_1 = """+s13+"""</span>
        <span>Possibility of Block 0_2 = """+s14+"""</span>
        <span>Possibility of Block 1_0 = """+s15+"""</span>
        <span>Possibility of Block 1_1 = """+s16+"""</span>
        <span>Possibility of Block 1_2 = """+s17+"""</span>
        <span>Possibility of Block 2_0 = """+s18+"""</span>
        <span>Possibility of Block 2_1 = """+s19+"""</span>
        <span>Possibility of Block 2_2 = """+s20+"""</span>
        <br/>
        </div>
        <div class="title">
        <span>Heat Map - Attention of Model (RBG PATTERN)</span>
        </div>
        <div class="text">
        <br/>
        <div class="graph">
        <img src=\""""+heat_map1+"""\" width="800px" height="200px" />
        </div>
        <br/>
        </div>
        <div class="title">
        <span>Heat Map - Attention of Model (BINARY PATTERN)</span>
        </div>
        <div class="text">
        <br/>
        <div class="graph">
        <img src=\""""+heat_map2+"""\" width="800px" height="200px" />
        </div>
        <br/>
        </div>
        <div class="title">
        <span>Probability Sharing - Consider the neighbor node information</span>
        </div>
        <div class="text">
        <span class="discribe">
        <br/>
        Consider the whole image as a graph just as we do in GNN<br/>
        After a few iteration, we achieve local convergence <br/>
        <br/>
        </span>
        <div class="graph">
        <img src="gcn.gif" width="300px" height="300px" border="2 solid red"/>
        <div class="prob">
        <span>Average Possibility Of RBG PATTERN ~ """+s21+"""</span>
        <br/>
        <br/>
        <br/>
        <span>Average Possibility Of BINARY PATTERN ~ """+s22+"""</span>
        </div>
        </div>
        <br/>
        </div>
        <div class="title">
        <span>Node Voting - Combine different pattern</span>
        </div>
        <div class="text">
        <span class="discribe">
        <br/>
        The ultimate outcome is the combination of different prediction models and a trained threshold.<br/>
        <br/>
        </span>
        <div class="graph">
        <div class="prob">
        <span>Prediction Result: """+s23+"""</span>
        <span class="note">"""+s24+"""</span>
        <br/>
        <br/>
        </div>
        </div>
        </div>
        </div>
        </body>
        <style>
        .main {
        max-width: 1000px;
        display: flex;
        flex-direction: column;
        align-items: center;
        }
        .head {
        width: 100%;
        height: 50px;
        display: flex;
        align-content: center;
        align-items: center;
        justify-content: center;
        color: black;
        }
        .head_content {
        font-weight: bold;
        font-size: 28px;
        }
        .label{
        display: flex;
        width:100%;
        background-color: #f9f9f9;
        }
        .labelbox{
        display:flex ;
        width:50%;
        justify-content: center;
        font-weight: bold;
        }
        .imgcontainer {
        width: 100%;
        display: flex;
        background-color: #f9f9f9;
        }
        .input {
        width: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        }
        .crop {
        width: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        }
        .block9{
        display: flex;
        height:400px;
        width:400px;
        flex-direction: row;
        align-items: center;
        }
        .patchs{
        width:100%;
        height:100%;
        display: flex;
        flex-direction: column;
        justify-content: space-around;
        }
        .patch{
        display: flex;
        flex-direction: column;
        align-content: center;
        justify-content: center;
        align-items: center;
        }
        .title{
        width:100%;
        height:50px;
        font-weight: bold;
        font-size:18px;
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: #ececec;
        }
        .text{
        display: flex;
        width:100%;
        /*height: 350px;*/
        background-color: #f9f9f9;
        text-align:center;
        justify-content: center;
        line-height: 1.5;
        align-items: center;
        flex-direction: column;
        }
        .discribe{
        color: #9a9a9a;
        }
        .graph{
        width:100%;
        display: flex;
        flex-direction: row;
        justify-content: space-evenly;
        align-items: center;
        }
        .prob{
        display: flex;
        height: 100%;
        flex-direction: column;
        }
        .note{
        font-size: 5px;
        }
        </style>
        </html>
        """
        f.write(message)
        f.close()
    #创建网页
    createweb(s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,newpath)

    time2 = time.time()
    delta = round(time2-time1,2)
    # print('总耗时',delta,'秒')

if __name__ == '__main__':
    args = parse_option()
    id = args.ids
    # input_path = "dataset/11/test5.jpg"
    input_path = args.imgpath
    run(id,input_path)
