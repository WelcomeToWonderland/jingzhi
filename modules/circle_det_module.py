import cv2
import numpy as np
import time
import os
from alive_progress import alive_bar
import random
import networkx as nx
import math
import shutil
import sys
import torch
from torchvision.transforms import ToTensor, ToPILImage
import json
from scipy.spatial import distance


def remove_tiny(image):
    """
    输入：背景为黑，目标为白的二值图像
    处理：将其中细小物体使用黑色覆盖
    输出：处理后的二值图像
    """
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        # 将面积小于50的轮廓填充为黑色
        if area < 50:
            cv2.fillPoly(image, [contour], 0)
    return image

    
def merge_circles(circles):
    """
    输入格式：（圆心x坐标，圆心y坐标，半径）
    """
    def cover(c1, c2):
        # 当两个圆的圆形和半径都相近时，融合为一个圆
        x1, y1, r1 = c1
        x2, y2, r2 = c2
        # 计算两个圆心的距离
        d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # 计算半径的绝对差
        dr = abs(r1 - r2)
        # 两者都小于10时，认为两个圆是同一个圆
        if d < 10 and dr < 10:
            # 计算新的圆心
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            # 计算新的半径
            r = (r1 + r2) / 2
            return True, (x, y, r)
        return False, None

    continue_flag = True
    while continue_flag:
        continue_flag = False
        break_flag = False
        for i in range(len(circles)):
            for j in range(len(circles)):
                if j == i:
                    continue
                is_cover, new_circle = cover(circles[i], circles[j])
                if is_cover:
                    break_flag = True
                    circles[i], circles[j] = new_circle, []
                    continue_flag = True
            if break_flag:
                break
        circles = [circle for circle in circles if circle]


    # for i in range(len(circles)):
    #     for j in range(len(circles)):
    #         if j == i or circles[j] == []:
    #             continue
    #         is_cover, new_circle = cover(circles[i], circles[j])
    #         if is_cover:
    #             circles[i], circles[j] = new_circle, []
    # circles = [circle for circle in circles if circle]

    return circles

# 圆形检测
def circle_det(img):
    circles = []
    mode = cv2.RETR_LIST
    contours, _ = cv2.findContours(img, mode=mode, method=cv2.CHAIN_APPROX_SIMPLE)
    # 遍历轮廓
    for contour in contours:
        # 计算圆形度
        if abs(cv2.arcLength(contour, True)) < 1e-9:
            continue
        circularity = 4 * np.pi * cv2.contourArea(contour) / (cv2.arcLength(contour, True) ** 2)
        if circularity > 0.85:
            # 计算对应圆形和半径
            center, radius = cv2.minEnclosingCircle(contour)
            circles.append((center[0], center[1], radius))
    return circles  

# 构件圆形检测
def component_circle_det(img_path_list):
    """
    输入图像地址列表，调用line_det函数遍历检测图像直线，建立字典res存储检测结果，key为地址，value为检测结果
    """
    res = {}

    for img_path in img_path_list:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to load image {img_path}")
            continue
            
        final_circles = []

        img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel_size = 3
        eroded = cv2.erode(binary, np.ones((kernel_size, kernel_size), np.uint8), iterations=2)
        eroded = remove_tiny(eroded)
        dilated_1 = cv2.dilate(eroded, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
        eroded = dilated_1

        final_circles += circle_det(eroded)

        eroded_1 = cv2.erode(binary, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)

        final_circles += circle_det(eroded_1)

        final_circles = merge_circles(final_circles)
        final_circles = [circle for circle in final_circles if circle[2] > 3]
        final_circles = [[x / 2 for x in sublist] for sublist in final_circles]
        # final_circles = np.array(final_circles)/2/
        
        res[img_path] = final_circles

    return res
# 代码测试
def test_circle_line_det():
    log_root = '../logs'
    logname = '构件圆形检测demo展示'

    img_dir = '../data/精智demo展示案例备选二'
    img_path_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
    if not os.path.exists(os.path.join(log_root, logname)):
        os.makedirs(os.path.join(log_root, logname))

    res = component_circle_det(img_path_list)
    # print(res)

    for img_path, circles in res.items():
        baseneame = os.path.basename(img_path)
        prefix, suffix = os.path.splitext(baseneame)
        img = cv2.imread(img_path)
        img_red = img.copy()
        for circle in circles:
            cv2.circle(img_red, (int(circle[0]), int(circle[1])), int(circle[2]), (0, 0, 255), 2)
            color = [random.randint(0, 255) for _ in range(3)]
            cv2.circle(img, (int(circle[0]), int(circle[1])), int(circle[2]), color, 2)
        cv2.imwrite(os.path.join(log_root, logname, f'{prefix}_red{suffix}'), img_red)
        cv2.imwrite(os.path.join(log_root, logname, f'{prefix}{suffix}'), img)



if __name__ == "__main__":
    test_circle_line_det()