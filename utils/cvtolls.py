
# from skimage import measure
import cv2 as cv 
import numpy as np
import xml.etree.ElementTree as ET
import os 
from pathlib import Path
from tqdm import tqdm


##
##image:二值图像
##threshold_point:符合面积条件大小的阈值
# def remove_small_points(image,threshold_point):
#     img = cv2.imread(image, 0)       #输入的二值图像
#     img_label, num = measure.label(img, neighbors=8, return_num=True)#输出二值图像中所有的连通域
#     props = measure.regionprops(img_label)#输出连通域的属性，包括面积等
 
#     resMatrix = np.zeros(img_label.shape)
#     for i in range(1, len(props)):
#         if props[i].area > threshold_point:
#             tmp = (img_label == i + 1).astype(np.uint8)
#             resMatrix += tmp #组合所有符合条件的连通域
#     resMatrix *= 255
#     return resMatrix
 
# res = remove_small_points(image,threshold_point)
 


# # 加载图片
# img = cv.imread('image_name.png',0)
# # 灰度化
# img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# # 二值化
# ret, thresh = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
# # 寻找连通域
# num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity=8)

# # 计算平均面积
# areas = list()
# for i in range(num_labels):
#     areas.append(stats[i][-1])
#     print("轮廓%d的面积:%d" % (i, stats[i][-1]))

# area_avg = np.average(areas[1:-1])
# print("轮廓平均面积:", area_avg)

# # 筛选超过平均面积的连通域
# image_filtered = np.zeros_like(img)
# for (i, label) in enumerate(np.unique(labels)):
#     # 如果是背景，忽略
#     if label == 0:
#         continue
#     if stats[i][-1] > area_avg :
#         image_filtered[labels == i] = 255

# cv.imshow("image_filtered", image_filtered)
# cv.imshow("img", img)
# cv.waitKey()
# cv.destroyAllWindows()


"""
cut object from image by xml file
"""
def cutObject(xmldir,imgdir,savedir):
    xmldir = Path(xmldir)
    imgdir = Path(imgdir)
    savedir = Path(savedir)
    for file in tqdm(xmldir.glob('*.xml')):
        imgPath = imgdir.joinpath(str(file.stem)+'.jpg')
        if not imgPath.exists():
            continue
        image = cv.imread(str(imgPath))
        root = ET.parse(str(file)).getroot()
        for obj in root.iter('object'):
            xmlbox = obj.find('bndbox')
            xmin,xmax,ymin,ymax = (xmlbox.find('xmin').text,xmlbox.find('xmax').text,xmlbox.find('ymin').text,xmlbox.find('ymax').text)
            obj_img = image[int(ymin):int(ymax),int(xmin):int(xmax)]
            savePath = str(savedir/imgPath.name)
            cv.imwrite(savePath, obj_img)
        pass 
# cutObject('data/Normal_Insulators/labels','data/Normal_Insulators/images','data/rgbImages')





