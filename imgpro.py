import time
import cv2 as cv
import numpy as np
import os
import shutil



#step 1 对比度增强
def img_toclahe(path):
    # First read img as rgb-origin type
    img_origin = cv.imread(path, cv.IMREAD_COLOR)
    # cv.imshow("img_origin", img_origin)
    # cv.waitKey(0)
    # split img into r,g,b channel
    b, g, r = cv.split(img_origin)
    # CLAHE 对比限制自适应直方图均衡化  clipLimit：对比度限制值，默认为40.0;tileGridSize：分块大小，默认为Size(8, 8)
    clahe = cv.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    # print(clahe)
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    # 合并通道
    img_clahe = cv.merge([b, g, r])
    # cv.imshow("img_clahe", img_clahe)
    # cv.imshow("img_origin", img_origin)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return img_origin,img_clahe

#step 2 膨胀（局部最大值）
def img_dilate(img_clahe):
    # 矩形卷积核
    kernel = np.ones((5,5),np.uint8)
    # 圆型卷积核
    kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    # x型卷积核
    kernel3 = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
    img_dilate = cv.dilate(img_clahe,kernel,iterations = 1)
    img_dilate2 = cv.dilate(img_clahe,kernel2,iterations = 1)
    img_dilate3 = cv.dilate(img_clahe, kernel3, iterations=1)
    # cv.imshow("img_clahe",img_clahe)
    # cv.imshow("img_dilate_rectangle", img_dilate)
    # cv.imshow("img_dilate_ellipse", img_dilate2)
    # cv.imshow("img_dilate_xray", img_dilate3)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return img_clahe,img_dilate,img_dilate2,img_dilate3

#step 3 腐蚀（局部最小值）
def img_erode(img_clahe,img_dilate,img_dilate2,img_dilate3):
    # 矩形卷积核
    kernel = np.ones((5,5),np.uint8)
    # 圆型卷积核
    kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    # x型卷积核
    kernel3 = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
    img_erode = cv.erode(img_dilate,kernel,iterations = 1)
    img_erode2= cv.erode(img_dilate2,kernel2,iterations = 1)
    img_erode3 = cv.erode(img_dilate3, kernel3, iterations=1)
    # cv.imshow("img_clahe",img_clahe)
    # cv.imshow("img_erode_rectangle", img_erode)
    # cv.imshow("img_erode_ellipse", img_erode2)
    # cv.imshow("img_erode_xray", img_erode3)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return img_erode, img_erode2, img_erode3

#choice 1 闭运算（强调边缘信息）
def closes(img_clahe):
    img,img_dict,img_dict2, img_dict3 = img_dilate(img_clahe)
    img_erod,img_erod2, img_erod3 = img_erode(img,img_dict,img_dict2, img_dict3)
    return img_erod2

#choice 2 开运算（去掉局部信息）
def opens(img_clahe):
    def img_dilate(img_clahe,img_dilate,img_dilate2,img_dilate3):
        # 矩形卷积核
        kernel = np.ones((5, 5), np.uint8)
        # 圆型卷积核
        kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        # x型卷积核
        kernel3 = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
        img_dilate = cv.dilate(img_dilate, kernel, iterations=1)
        img_dilate2 = cv.dilate(img_dilate2, kernel2, iterations=1)
        img_dilate3 = cv.dilate(img_dilate3, kernel3, iterations=1)
        cv.imshow("img_clahe", img_clahe)
        cv.imshow("img_dilate_rectangle", img_dilate)
        cv.imshow("img_dilate_ellipse", img_dilate2)
        cv.imshow("img_dilate_xray", img_dilate3)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return img_clahe,img_dilate,img_dilate2,img_dilate3

    # step 3 腐蚀（局部最小值）
    def img_erode(img_clahe):
        # 矩形卷积核
        kernel = np.ones((5, 5), np.uint8)
        # 圆型卷积核
        kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        # x型卷积核
        kernel3 = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
        img_erode = cv.erode(img_clahe, kernel, iterations=1)
        img_erode2 = cv.erode(img_clahe, kernel2, iterations=1)
        img_erode3 = cv.erode(img_clahe, kernel3, iterations=1)
        cv.imshow("img_clahe", img_clahe)
        cv.imshow("img_erode_rectangle", img_erode)
        cv.imshow("img_erode_ellipse", img_erode2)
        cv.imshow("img_erode_xray", img_erode3)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return img_clahe, img_erode, img_erode2, img_erode3

    img_clahe, img_erode, img_erode2, img_erode3 = img_erode(img_clahe)
    img_dilate(img_clahe, img_erode, img_erode2, img_erode3)

# step 4 图片灰度化，二值化，融合化
def img_gray_binary(img_clahe,img_erod2,ctrl_num):
    # 图像的灰度化
    img_gray = cv.cvtColor(img_clahe, cv.COLOR_RGB2GRAY)
    img_gray2 = cv.cvtColor(img_erod2, cv.COLOR_RGB2GRAY)
    # 图像的二值化
    # #获取图像的像素值范围
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(img_gray)
    #　＃设定阈值为像素值的50%
    threshold = (min_val+max_val)/ctrl_num
    max_value = 255
    [_, img_bin] = cv.threshold(img_gray, threshold, max_value, cv.THRESH_BINARY)
    [_, img_bin2] = cv.threshold(img_gray2, threshold, max_value, cv.THRESH_BINARY)
    # 二值化+灰度化+融合化
    img1 = img_gray2
    img2 = img_bin
    rows, cols = img2.shape
    roi = img1[0:rows, 0:cols]
    ret, mask = cv.threshold(img2, 1, 255, cv.THRESH_BINARY)
    img_com = cv.bitwise_and(roi, roi, mask=mask)
    # cv.imshow("img_clahe", img_clahe)
    # cv.imshow("img_clahe_gray", img_gray)
    # cv.imshow("closes_gray2", img_gray2)
    # cv.imshow("closes", img_erod2)
    # cv.imshow("img_clahe_bin", img_bin)
    # cv.imshow("closes_bin2", img_bin2)
    # cv.imshow('img_com', img_com)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return img_com

# step 5 图片边缘检测
def img_laplician_canny(img_origin,img_clahe,img_com):
    # 原图直接边缘检测
    img_origin = cv.Canny(img_origin, 0, 100)
    # clahe直接边缘检测
    img_clahe = cv.Canny(img_clahe, 0, 100)
    # 进一步边缘检测 canny算法
    img_canny = cv.Canny(img_com,0,100)
    # laplacian算子检测
    img_Laplacian = cv.Laplacian(img_com,cv.CV_16S)
    img_Laplacian = cv.convertScaleAbs(img_Laplacian)
    # cv.imshow("img_origin", img_origin)
    # cv.imshow("img_clahe", img_clahe)
    # cv.imshow("img_origin_canny", img_origin)
    # cv.imshow("img_clahe_canny", img_clahe)
    # cv.imshow("img_com_canny", img_canny)
    # cv.imshow("img_com_laplacian", img_Laplacian)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return img_canny

# step 6 创建文件夹
def create_folder():
    res_folder1 = r'D:\datasets\dataset1\test'
    res_folder2 = r'D:\datasets\dataset1\train\best'
    res_folder3 = r'D:\datasets\dataset1\train\mid'
    res_folder4 = r'D:\datasets\dataset1\train\bad'
    res_folder5 = r'D:\datasets\dataset1\val\best'
    res_folder6 = r'D:\datasets\dataset1\val\mid'
    res_folder7 = r'D:\datasets\dataset1\val\bad'
    res_folder8 = r'D:\datasets\dataset2\test'
    res_folder9 = r'D:/datasets/dataset2/train/best'
    res_folder10 = r'D:\datasets\dataset2\train\mid'
    res_folder11 = r'D:\datasets\dataset2\train\bad'
    res_folder12 = r'D:\datasets\dataset2\val\best'
    res_folder13 = r'D:\datasets\dataset2\val\mid'
    res_folder14 = r'D:\datasets\dataset2\val\bad'
    names = locals()
    for i in names.keys():
        j = names['%s'%i]
        if os.path.exists(j):
            shutil.rmtree(j)
        os.makedirs(j)

# step 7 rename file
def img_rename(img_path,count,dirpath):
    oldname = img_path
    filename = str(count)+'.png'
    newname = os.path.join(dirpath,filename)
    # print(oldname,newname)
    # print('----------------------------------------------')
    os.rename(oldname,newname)
    return newname

# step 8 边缘检测图片合成
def img_to_com_canny(path,ctrl_num):
    img_origin,img_clahe = img_toclahe(path)
    img_erod2 = closes(img_clahe)
    img_com = img_gray_binary(img_clahe,img_erod2,ctrl_num)
    img_canny = img_laplician_canny(img_origin,img_clahe,img_com)
    # img_com_path = r'D:\datasets\dataset1'
    # img_canny_path = r'D:\datasets\dataset2'
    # img_com_path2 = os.path.join(img_com_path,filename)
    # img_canny_path2 = os.path.join(img_canny_path,filename)

    cv.imwrite(path, img_canny)
    # cv.imwrite(img_canny_path2, img_canny)

# step 9 生成dataset
def classiy_img(input_path,ctrl_num=3):
    for dirpath, dirnames, filenames in os.walk(input_path):
        if dirpath != r'D:\AI\swin-transformer\imgs':
            assert filenames, '未找到文件！'
            # print(dirpath)
            # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            triger = dirpath[-1]
            count = 0
            test_num = round(len(filenames)*0.1)
            train_num = round(len(filenames)*0.8)
            for filename in filenames:
                # print(filename,count)
                if count < test_num:
                    img_path = os.path.join(dirpath, filename)
                    img_path2 = img_rename(img_path,count,dirpath)
                    id = 'test'
                    img_to_com_canny(img_path2, count, triger, id,ctrl_num)
                    count +=1
                elif count < (test_num+train_num):
                    img_path = os.path.join(dirpath, filename)
                    img_path2 = img_rename(img_path, count, dirpath)
                    id = 'train'
                    img_to_com_canny(img_path2, count, triger, id,ctrl_num)
                    count += 1
                elif count < len(filenames):
                    img_path = os.path.join(dirpath, filename)
                    img_path2 = img_rename(img_path, count, dirpath)
                    id = 'val'
                    img_to_com_canny(img_path2, count, triger, id,ctrl_num)
                    count += 1

# 图片归一化
def normlize(path):
    image = cv.imread(path)
    cv.imshow("input", image)
    result = np.zeros(image.shape, dtype=np.float32)
    cv.normalize(image, result, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    print(result)
    cv.imshow("norm", np.uint8(result * 127))
    cv.waitKey(0)
    cv.destroyAllWindows()

# 图片转灰度
path = './dataset/ppt/pp1.jpg'
img = cv.imread(path)

cv.imshow("img_origin", img)
cv.waitKey(0)
img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# img_gray = cv.imread(img_clahe,cv.IMREAD_GRAYSCALE)
cv.imshow("img_origin", img_gray)
cv.waitKey(0)
img_canny = cv.Canny(img_gray,0,100)
cv.imshow("img_canny", img_canny)
cv.waitKey(0)
kernel = np.ones((5,5), np.uint8)
erosion = cv.dilate(img_canny, kernel)
cv.imshow("erosion", erosion)
cv.waitKey(0)
h, w = img_gray.shape
# print(h,w)
total_pixel = h*w
pixel_percent = 0.005*total_pixel
# print(pixel_percent)
s_list = []
s_dict = {}
for row in range(h):  # 对图中所有的像素点进行遍历
    for col in range(w):
        s = img_gray[row, col]
        if s in s_list:
            temp = s_dict['%s'%s]
            s_dict['%s' % s] = temp+1
        else:
            s_list.append(s)
            s_dict['%s'%s] = 1
# print(s_list)
# print(max(s_list))
# print(min(s_list))
# print(s_dict['111'])
# print(sorted(s_dict))
for k in list(s_dict.keys()):
    if s_dict[k]< pixel_percent:
        del s_dict[k]
# print(sorted(s_dict))
bina_list = []
for j in sorted(s_dict):
    if not bina_list:
        bina_list.append(j)
    else:
        if (int(j)-int(bina_list[-1]))>20:
            bina_list.append(j)
print(bina_list)

hold = 0
pix = 0
for i in bina_list:
    hold +=(int(i)*s_dict[i])
    pix +=s_dict[i]
hold /=pix
print(round(hold))
ret, binary = cv.threshold(img_gray, round(hold), 255, cv.THRESH_TOZERO)# 自定义阈值为150,大于150的是黑色 小于的是白色
print("阈值：%s" % ret)
cv.imshow("自定义反色", binary)
cv.waitKey(0)
# kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
# bin_clo = cv.dilate(binary, kernel2, iterations=2)
dst=np.zeros((h,w,1),np.uint8)
for i in range(h):
    for j in range(w):
        dst[i,j]=255-erosion[i,j]
num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(dst, connectivity=4)
output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
for i in range(1, num_labels):

    mask = labels == i
    output[:, :, 0][mask] = np.random.randint(0, 255)
    output[:, :, 1][mask] = np.random.randint(0, 255)
    output[:, :, 2][mask] = np.random.randint(0, 255)
cv.imshow('oginal', output)
cv.waitKey()
cv.destroyAllWindows()

#if __name__ == '__main__':
#     time1 = time.time()
#     curtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#     print(curtime,'-----','开始处理图片。。。')
#     #在这里设置dataset的输出位置
#     print(curtime, '-----', '输出路径为：D:\datasets')
#     create_folder()
#     #在这里置换文件夹位置，改为自己的imgs（包含0,1,2）的原始图片地址
#     input_path = r'D:\AI\swin-transformer\imgs'
#     print(curtime, '-----', '读取路径为：',input_path)
#     #阈值阀,默认为3,调节图片二值化效果用的，如果生的dataset信息过少（黑色部分太多）增大阈值,反之亦然
#     # ctrl_num = 自定义数字
#     # classiy_img(input_path,ctrl_num)
#     classiy_img(input_path)
#     time2 = time.time()
#     delta = str(round(time2-time1,2))
#     print(curtime, '-----', 'dataset已生成')
