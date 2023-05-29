import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
import cv2
import csv
import os
from os import listdir
import math
from skimage.filters import prewitt_h,prewitt_v

file = open('stl10.csv', 'a', newline='')
writer = csv.writer(file)
# headerList = ["Mean", "Median", "Variance", "Standard Deviation","Skewness","Prewitt X","Prewitt Y","Sobel X","Sobel Y","Canny Edge","label"]
# dw = csv.DictWriter(file, delimiter=',',fieldnames=headerList)
# dw.writeheader()
folder_airplane = "./img/1"
folder_car = "./img/3"


images_names=[]
images_list=[]
for image in os.listdir(folder_airplane):
    images_names.append(image)
    img_original = cv2.imread("./img/1/"+image)
    images_list.append(cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY))


print(len(images_list))
for image in os.listdir(folder_car):
    images_names.append(image)
    img_original = cv2.imread("./img/3/"+image)
    images_list.append(cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY))


# plt.imshow(images_list[0])
# plt.show()

print(len(images_list))

for img in range(500,1000,1):
    mean=0
    variance=0
    pixels_list=[]
    for i in range(0,images_list[img].shape[0]):
        for j in range(0,images_list[img].shape[1]):
            mean=mean+images_list[img][i][j]
            pixels_list.append(images_list[img][i][j])
    mean=mean/images_list[img].size
    pixels_list.sort()
    median_index=math.floor(len(pixels_list)/2)
    median=pixels_list[median_index]
    for i in range(0,images_list[img].shape[0]):
        for j in range(0,images_list[img].shape[1]):
            variance= variance+ math.pow(images_list[img][i][j]-mean ,2)
    variance=variance/images_list[img].size
    standard_deviation=math.sqrt(variance)
    skewness=3*((mean-median)/standard_deviation)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(images_list[img],(3,3),0)
    sobelx = cv2.Sobel(src=img_gaussian, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5).flatten().max()
    sobely = cv2.Sobel(src=img_gaussian, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5).flatten().mean()
    sobelxy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5).flatten().mean()
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx).flatten().mean()
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely).flatten().mean()
    cannyedge = cv2.Canny(image=images_list[img], threshold1= 100,threshold2= 200)
    canny_edge=cannyedge.flatten().mean()
    # print("mean value of image",img+1,"is:",mean, "and median is:",median)
    data_csv=[mean,median,variance,standard_deviation,skewness,img_prewittx,img_prewitty,sobelx,sobely,canny_edge,0]
    writer.writerow(data_csv)

file.close()