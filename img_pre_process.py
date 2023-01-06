import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
# 输入图片，判断空腔面积，返回置信概率
from matplotlib.pyplot import loglog


def get_conf():
    path = './dataset/sp9/2_2.jpg'
    img = cv.imread(path)
    # cv.imshow("img_origin", img)
    # cv.waitKey(0)
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # img_gray = cv.imread(img_clahe,cv.IMREAD_GRAYSCALE)
    # cv.imshow("img_origin", img_gray)
    # cv.waitKey(0)
    # cv.imwrite('./dataset/ppt/img_gray2.png', img_gray)
    img_canny = cv.Canny(img_gray,0,100)
    # cv.imshow("img_canny", img_canny)
    # cv.waitKey(0)
    # cv.imwrite('./dataset/ppt/img_canny2.png', img_canny)
    kernel = np.ones((5,5), np.uint8)
    dilate = cv.dilate(img_canny, kernel)
    # cv.imwrite('./dataset/ppt/erosion21.png', dilate)
    dilate2 = cv.dilate(dilate, kernel)
    # cv.imwrite('./dataset/ppt/ierosion22.png', dilate2)
    dilate3 = cv.dilate(dilate2, kernel)
    # cv.imwrite('./dataset/ppt/erosion32.png', dilate3)
    # cv.imshow("dilate", dilate)
    # cv.waitKey(0)
    # cv.imshow("dilate2", dilate2)
    # cv.waitKey(0)
    # cv.imshow("dilate3", dilate3)
    # cv.waitKey(0)
    h, w = img_gray.shape
    dst=np.zeros((h,w,1),np.uint8)
    for i in range(h):
        for j in range(w):
            dst[i,j]=255-dilate3[i,j]
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(dst, connectivity=8)
    output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    # for i in range(1, num_labels):
    #     mask = labels == i
    #     output[:, :, 0][mask] = np.random.randint(0, 255)
    #     output[:, :, 1][mask] = np.random.randint(0, 255)
    #     output[:, :, 2][mask] = np.random.randint(0, 255)
    cv.imwrite('./dataset/output2.png', dst)
    # cv.imshow('output', output)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    s_list = []
    s_dict = {}
    # cv.imwrite('./dataset/ppt/dst2.png', dst)
    # cv.imshow('dst', dst)
    # cv.waitKey(0)
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
    print(max_area)
    percent = round(max_area/total,3)
    print(percent)
    if percent<=0.1:
        prob = 1
    else:
        prob = round(-math.log(percent,10),3) # 使用math中的log函数生成对应x的值
    print(prob)
    return percent,prob

x_1 ,y_1= get_conf()
plt.title('Piecewise-confidence function')
# arange函数的含义是[start,end,step)
x = np.arange(0.0000001,1,0.001) # 注意区间，因为logx中的x>0，所以这里的区间设置成[0.0000001,5)
y = []
for i in x:
    if i<=0.1:
        y.append(1)
    else:
        temp = -math.log(i,10) # 使用math中的log函数生成对应x的值
        y.append(temp) # 放入到数组y中
plt.plot(x,y)  #
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.set_aspect(1)
plt.scatter(x_1, y_1, s=125, c='r')
if x_1<=0.1:
    print(x_1,y_1)
    plt.text(x_1+0.3, y_1-0.05, 'Confidence='+str(y_1), ha='center', va='bottom', fontsize=10.5)
    plt.plot([x_1, x_1], [0, y_1], c='b', linestyle='--')
else:
    print(x_1, y_1)
    plt.text(x_1+0.2, y_1+0.05, 'Confidence='+str(y_1), ha='center', va='bottom', fontsize=10.5)
    plt.plot([0, x_1], [y_1, y_1], c='b', linestyle='--')
    plt.plot([x_1, x_1], [y_1, 0], c='b', linestyle='--')
plt.grid(True)
# plt.show()
plt.savefig('./dataset/mat1.png')
