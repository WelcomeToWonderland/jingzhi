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


# 霍夫检测相关中间处理函数
def remove_similar_lines(lines, distance_threshold=3):
    """
    过滤线段列表，去除相互靠近的线段。
    
    参数:
    - lines: 输入的线段列表，每个线段表示为 [[x1, y1, x2, y2], ...]
    - distance_threshold: 距离阈值，小于等于该距离的线段将被视为重复的（默认为3）

    返回:
    - 经过过滤的线段列表

    同时也达到了np.squeeze的作用
    """
    line_list = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        complicated = False
        
        for l in line_list:
            if math.sqrt((l[0] - x1)**2 + (l[1] - y1)**2 + (l[2] - x2)**2 + (l[3] - y2)**2) <= distance_threshold:
                complicated = True
                break
        
        if not complicated:
            line_list.append([x1, y1, x2, y2])
    
    return line_list

def cover(l1, l2):
    l1h = True
    l2h = True
    if abs(l1[0] - l1[2]) <= 5:
        l1h = False
    if abs(l2[0] - l2[2]) <= 5:
        l2h = False
    if l1h != l2h:
        return False, [-1, -1, -1, -1]
    if l1h:
        # 水平线
        # if abs(l1[1] - l2[1]) <= 10:
        if abs(l1[1] - l2[1]) <= 5:
            l1[0], l1[2] = min(l1[0], l1[2]), max(l1[0], l1[2])
            l2[0], l2[2] = min(l2[0], l2[2]), max(l2[0], l2[2])
            if l1[0] <= l2[0] <= l1[2]:
                return True, [l1[0], (l1[1] + l2[1]) // 2, max(l1[2], l2[2]), (l1[1] + l2[1]) // 2]
            elif l2[0] > l1[2]:
                return False, [-1, -1, -1, -1]
            else:
                if l1[0] <= l2[2]:
                    return True, [l2[0], (l1[1] + l2[1]) // 2, max(l1[2], l2[2]), (l1[1] + l2[1]) // 2]
                else:
                    return False, [-1, -1, -1, -1]
        else:
            return False, [-1, -1, -1, -1]
    else:
        # if abs(l1[0] - l2[0]) <= 10:
        if abs(l1[0] - l2[0]) <= 5:
            l1[1], l1[3] = min(l1[1], l1[3]), max(l1[1], l1[3])
            l2[1], l2[3] = min(l2[1], l2[3]), max(l2[1], l2[3])
            if l1[1] <= l2[1] <= l1[3]:
                return True, [(l1[0] + l2[0]) // 2, l1[1], (l1[0] + l2[0]) // 2, max(l1[3], l2[3])]
            elif l2[1] > l1[3]:
                return False, [-1, -1, -1, -1]
            else:
                if l1[1] <= l2[3]:
                    return True, [(l1[0] + l2[0]) // 2, l2[1], (l1[0] + l2[0]) // 2, max(l1[3], l2[3])]
                else:
                    return False, [-1, -1, -1, -1]
        else:
            return False, [-1, -1, -1, -1]

def merge_line(line_list):
    continue_flag = True
    while continue_flag:
        continue_flag = False
        break_flag = False
        for i in range(len(line_list)):
            for j in range(len(line_list)):
                if j == i:
                    continue
                is_cover, new_line = cover(line_list[i], line_list[j])
                if is_cover:
                    break_flag = True
                    line_list[i], line_list[j] = new_line, []
                    continue_flag = True
            if break_flag:
                break
        line_list = [line for line in line_list if line]
    return line_list


# 师清，直线检测（所有类型）
"""
修改
filter_lines函数
取消“长度筛选”和“角度筛选”
"""
def is_intersecting(line1, line2):
    """判断两条线段是否相交
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    if max(x1, x2) >= min(x3, x4) and min(x1, x2) <= max(x3, x4):  # 判断x轴上的投影是否有重叠
        if max(y1, y2) >= min(y3, y4) and min(y1, y2) <= max(y3, y4):  # 判断y轴上的投影是否有重叠
            return True

    return False


def segments_distance(line1, line2):
    """ 两条线段的最小距离：
        The closest distance between two segments is either 0 if they intersect,
        or the distance from one of the segments' end points to the other segment
  """
    x11, y11, x12, y12 = line1
    x21, y21, x22, y22 = line2
    if is_intersecting(line1, line2):
        return 0

    # try each of the 4 vertices w/the other segment
    distances = []
    distances.append(point_segment_distance(x11, y11, x21, y21, x22, y22))
    distances.append(point_segment_distance(x12, y12, x21, y21, x22, y22))
    distances.append(point_segment_distance(x21, y21, x11, y11, x12, y12))
    distances.append(point_segment_distance(x22, y22, x11, y11, x12, y12))

    return min(distances)


def point_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:  # the segment's just a point
        return math.hypot(px - x1, py - y1)  #欧几里得范数

    # Calculate the t that minimizes the distance.
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

    # See if this represents one of the segment's
    # end points or a point in the middle.
    if t < 0:
        dx = px - x1
        dy = py - y1
    elif t > 1:
        dx = px - x2
        dy = py - y2
    else:
        near_x = x1 + t * dx
        near_y = y1 + t * dy
        dx = px - near_x
        dy = py - near_y

    return math.hypot(dx, dy)


def is_overlap(line1, line2, epsilon1=10, epsilon2=5):
    """判断两条线段是否重叠
    1. 距离小于某个设定阈值
    2. 斜率之差小于某个阈值
    """
    flag1 = calculate_angle(line1, line2) < epsilon1
    flag2 = segments_distance(line1, line2) < epsilon2

    return (flag1 and flag2)


def calculate_angle(line1, line2):
    """计算两个线段的夹角"""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    v1 = (x2 - x1, y2 - y1)
    v2 = (x4 - x3, y4 - y3)
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    norm_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    norm_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    angle_cos = abs(dot_product / (norm_v1 * norm_v2)) #夹角是锐角
    if angle_cos > 1:
        # print(angle_cos)
        angle_cos = 1
    angle_rad = math.acos(angle_cos)

    return math.degrees(angle_rad)


def calculate_len(line):
    """计算线段长度"""
    x1, y1, x2, y2 = line

    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def filter_lines(lines):
    """
    过滤线段：
    1. 去除重复线段
    2. 太短线段
    """

    line_list1 = []  #保存水平线和垂直线
    line_list2 = []  #保留倾斜线
    for idx, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        reptition = False
        for l in line_list1:
            if math.sqrt((l[0] - x1)**2 + (l[1] - y1)**2 + (l[2] - x2)**2 + (l[3] - y2)**2) <= 3:  #判断重复
                reptition = True
                break
        for l in line_list2:
            if math.sqrt((l[0] - x1)**2 + (l[1] - y1)**2 + (l[2] - x2)**2 + (l[3] - y2)**2) <= 3:  #判断重复
                reptition = True
                break
        if reptition:
            continue

        k = (y2 - y1) / (x2 - x1 + 1e-7)  #计算斜率
        if abs(k) > 0.1 and abs(k) < 10:
            line_list2.append([int(x1), int(y1), int(x2), int(y2)])  #倾斜线
        else:
            line_list1.append([int(x1), int(y1), int(x2), int(y2)])  #水平或垂直线

    # # 可能去除需要的斜线，所以注释了
    line_list2_filter = line_list2 
    # 角度过滤
    # line_list2_filter = []
    # for line2 in line_list2:  #如果斜线和水平或者垂直直线相交,夹角小于20，则去除此斜线
    #     removed = False
    #     for line1 in line_list1:
    #         if is_intersecting(line1, line2) and calculate_angle(line1, line2) < 20:  #线段相交且夹角小于20，过滤
    #             removed = True
    #             break
    #     if not removed:
    #         line_list2_filter.append(line2)
 
    # 长度过滤
    #如果长度太小，则去除
    # maxlen = calculate_len(max(line_list1, key=calculate_len))
    # line_list1_filter = list(filter(lambda line: calculate_len(line) > 0.1 * maxlen, line_list1))  #如果长度<0.1*最大长度，则过滤
    # line_list2_filter = list(filter(lambda line: calculate_len(line) > 0.1 * maxlen, line_list2_filter))  #如果长度<0.1*最大长度，则过滤
    # line_list1_filter = list(filter(lambda line: calculate_len(line) > 0.01 * maxlen, line_list1))  #如果长度<0.1*最大长度，则过滤
    # line_list2_filter = list(filter(lambda line: calculate_len(line) > 0.01 * maxlen, line_list2_filter))  #如果长度<0.1*最大长度，则过滤

    line_list1_filter = line_list1

    return line_list1_filter, line_list2_filter


def merge_lines_by_contour(lines, shape):
    """用轮廓和联通区分析的方法合并"""
    lines_merged = []

    #水平方向
    src = np.zeros(shape, np.uint8)
    for line in lines:
        x1, y1, x2, y2 = line  #两点确定一条直线，这里就是通过遍历得到的两个点的数据 （x1,y1）(x2,y2)
        k = (y2 - y1) / (x2 - x1 + 1e-7)
        if abs(k) < 0.1:
            cv2.line(src, (x1, y1), (x2, y2), (0, 0, 255), 3)  #在原图上画线

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    binar = (gray > 70).astype(np.uint8)

    horizonal_kernel1 = np.ones((3, 3), np.uint8)
    binar = cv2.morphologyEx(binar, cv2.MORPH_CLOSE, horizonal_kernel1)  #形态学操作
    horizonal_kernel2 = np.ones((1, 7), np.uint8)
    binar = cv2.erode(binar, kernel=horizonal_kernel2)
    contours, hierarchy = cv2.findContours(binar, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        line = (int(x), int(y + h / 2), int(x + w), int(y + h / 2))
        lines_merged.append(line)

    #垂直方向
    src = np.zeros(shape).astype(np.uint8)
    for line in lines:
        x1, y1, x2, y2 = line  #两点确定一条直线，这里就是通过遍历得到的两个点的数据 （x1,y1）(x2,y2)
        k = (y2 - y1) / (x2 - x1 + 1e-7)
        if abs(k) > 10:
            cv2.line(src, (x1, y1), (x2, y2), (0, 0, 255), 3)  #在原图上画线
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    binar = (gray > 70).astype(np.uint8)

    vertical_kernel1 = np.ones((3, 3), np.uint8)
    binar = cv2.morphologyEx(binar, cv2.MORPH_CLOSE, vertical_kernel1)  #形态学操作
    vertical_kernel2 = np.ones((7, 1), np.uint8)
    binar = cv2.erode(binar, kernel=vertical_kernel2)
    contours, hierarchy = cv2.findContours(binar, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        y = y + 5
        h = h - 10
        line = (int(x + w / 2), int(y), int(x + w / 2), int(y + h))
        lines_merged.append(line)

    return lines_merged


def point_project_on_line(a, b, p):
    """将p点投影到ab上"""
    a = np.array(a)
    b = np.array(b)
    p = np.array(p)
    ap = p - a
    ab = b - a
    result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
    return result


def merge_lines_by_overlap(lines):
    lines_merged = []

    for line1 in lines:
        for line2 in lines_merged:
            if is_overlap(line1, line2):
                #将重叠的line1和line2融合成新的new_line
                lines_merged.remove(line2)
                x1, y1, x2, y2 = line2
                x1, y1 = point_project_on_line((line1[0], line1[1]), (line1[2], line1[3]), (x1, y1))
                x2, y2 = point_project_on_line((line1[0], line1[1]), (line1[2], line1[3]), (x2, y2))
                x3, y3, x4, y4 = line1
                #从四个点中融合成一条线段,距离最远的
                pts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                distances = distance.cdist(pts, pts, 'euclidean')
                maxarg = np.unravel_index(distances.argmax(), distances.shape)
                line1 = int(pts[maxarg[0]][0]),int(pts[maxarg[0]][1]), int(pts[maxarg[1]][0]), int(pts[maxarg[1]][1])
                break

        lines_merged.append(line1)

    return lines_merged


def line_det(img, shape, imgname, method="ensemble"):
    """
    检测图像中的线段
    :param img: 处理完的图像像素矩阵（rgb形式），直接用于检测
    :param shape: 图像的形状
    :param imgname: 图像的名称
    :param method: 线段检测的方法，可选 "lsd", "fld", "hough", "hough_p", "ensemble"
    :return: 检测到的线段列表
    """
    lines = []

    if method == "lsd":
        # LineSegmentDetector
        lsd = cv2.createLineSegmentDetector(refine=cv2.LSD_REFINE_NONE, ang_th=30)
        lines, width, prec, nfa = lsd.detect(img)
    elif method == "fld":
        fld = cv2.ximgproc.createFastLineDetector(distance_threshold=10, canny_aperture_size=7, do_merge=True)
        lines = fld.detect(img)
    elif method == "hough":
        # Hough变换只能得到直线
        lines = cv2.HoughLines(img, rho=1, theta=np.pi / 2, threshold=100, srn=0, stn=5, min_theta=0, max_theta=1)
        if lines is not None:
            def polar2cartesian(lines):
                for rho, theta in lines[0]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 100 * (-b))
                    y1 = int(y0 + 100 * (a))
                    x2 = int(x0 - 100 * (-b))
                    y2 = int(y0 - 100 * (a))
                return (x0, y0, x2, y2)
            lines = list(map(polar2cartesian, lines))
    elif method == "hough_p":
        # only detect vertical and horizontal lines
        lines = cv2.HoughLinesP(img, rho=1, theta=np.pi / 180, threshold=300, minLineLength=50, maxLineGap=5)
    elif method == "ensemble":
        lsd = cv2.createLineSegmentDetector(refine=cv2.LSD_REFINE_STD, scale=1.0, ang_th=35, quant=3)
        lines_lsd, width, prec, nfa = lsd.detect(img)
        # fld = cv2.ximgproc.createFastLineDetector(distance_threshold=10, canny_th1=30, canny_th2=100, canny_aperture_size=7, do_merge=True)
        # lines_fld = fld.detect(img)
        lines_hough_p = cv2.HoughLinesP(img, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=5)

        # 检查检测结果是否为空，并过滤掉空结果
        valid_lines = []
        if lines_lsd is not None and len(lines_lsd) > 0:
            valid_lines.append(lines_lsd)
        # if lines_fld is not None and len(lines_fld) > 0:
        #     valid_lines.append(lines_fld)
        if lines_hough_p is not None and len(lines_hough_p) > 0:
            valid_lines.append(lines_hough_p)

        if valid_lines:
            lines = np.concatenate(valid_lines, axis=0)
        else:
            lines = []

    else:
        print(f"Error: Invalid line detection method '{method}' for image {imgname}!")
        return None

    if lines is None or len(lines) == 0:
        print(f"No lines detected for image {imgname} using method {method}.")
        return []

    line_list1, line_list2 = filter_lines(lines)

    lines_res1 = merge_lines_by_contour(line_list1, shape)
    lines_res2 = merge_lines_by_overlap(line_list2)

    return lines_res1 + lines_res2


# 预处理函数
def process_image(img):
    """
    处理输入图像，返回处理后的图像。
    
    参数:
    - img: 输入图像的像素矩阵
    
    返回:
    - 处理后的图像像素矩阵
    """
    
    # 调整图像大小
    img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
    
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 腐蚀操作
    kernel_size = 3
    eroded = cv2.erode(binary, np.ones((kernel_size, kernel_size), np.uint8), iterations=2)
    # eroded = remove_tiny(eroded)
    
    # 膨胀操作
    dilated_1 = cv2.dilate(eroded, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    eroded = dilated_1
    
    # 查找轮廓并处理
    mode = cv2.RETR_LIST
    contours, _ = cv2.findContours(eroded, mode=mode, method=cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if 220 < area < 300:
            cv2.fillPoly(eroded, [contour], 0)
    arrows_removed = eroded
    # arrows_removed = remove_tiny(arrows_removed)
    
    # # 再次查找轮廓并处理
    # mode = cv2.RETR_EXTERNAL
    # contours, _ = cv2.findContours(arrows_removed, mode=mode, method=cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     if area < 3000:
    #         cv2.fillPoly(arrows_removed, [contour], 0)
    # outside_removed = arrows_removed
    
    # # 最后一次查找轮廓并处理
    # mode = cv2.RETR_LIST
    # contours, _ = cv2.findContours(outside_removed, mode=mode, method=cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours:
    #     if abs(cv2.arcLength(contour, True)) < 1e-9:
    #         continue
    #     circularity = 4 * np.pi * cv2.contourArea(contour) / (cv2.arcLength(contour, True) ** 2)
    #     if circularity > 0.85:
    #         cv2.fillPoly(outside_removed, [contour], 0)
    
    # # 再次进行腐蚀操作
    # eroded_1 = cv2.erode(binary, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    # mode = cv2.RETR_LIST
    # contours, _ = cv2.findContours(eroded_1, mode=mode, method=cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours:
    #     if abs(cv2.arcLength(contour, True)) < 1e-9:
    #         continue
    #     circularity = 4 * np.pi * cv2.contourArea(contour) / (cv2.arcLength(contour, True) ** 2)
    #     if circularity > 0.85:
    #         cv2.fillPoly(outside_removed, [contour], 0)
    # circles_removed = outside_removed


    kernel_size = 3
    eroded = cv2.erode(arrows_removed, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    
    return eroded
    
def test_process_image():
    log_root = '/home/chenzhuofan/project_que/pipeline_jingzhi/logs'
    logname = '构件直线检测-图像预处理测试'

    img_dir = '/home/chenzhuofan/project_que/pipeline_jingzhi/data/精智demo展示案例备选二'
    img_path_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
    if not os.path.exists(os.path.join(log_root, logname)):
        os.makedirs(os.path.join(log_root, logname))

    for img_path in img_path_list:
        img = cv2.imread(img_path)
        processed_img = process_image(img)
        cv2.imwrite(os.path.join(log_root, logname, os.path.basename(img_path)), processed_img)


# 检测函数
def line_det_hough(img):
    """
    检测图像中的线段
    :param img: 处理完的图像像素矩阵（rgb形式），直接用于检测
    :param shape: 图像的形状
    :param imgname: 图像的名称
    :param method: 线段检测的方法，可选 "lsd", "fld", "hough", "hough_p", "ensemble"
    :return: 检测到的线段列表
    """
    lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/2, threshold=10, minLineLength=30, maxLineGap=10)
    lines = remove_similar_lines(lines)
    lines = merge_line(lines)
    return lines


def line_det_component(img, shape, imgname, method="ensemble"):
    """
    检测图像中的线段
    :param img: 处理完的图像像素矩阵（rgb形式），直接用于检测
    :param shape: 图像的形状
    :param imgname: 图像的名称
    :param method: 线段检测的方法，可选 "lsd", "fld", "hough", "hough_p", "ensemble"
    :return: 检测到的线段列表
    """
    lines = []

    if method == "lsd":
        # LineSegmentDetector
        lsd = cv2.createLineSegmentDetector(refine=cv2.LSD_REFINE_NONE, ang_th=30)
        lines, width, prec, nfa = lsd.detect(img)
    elif method == "fld":
        fld = cv2.ximgproc.createFastLineDetector(distance_threshold=10, canny_aperture_size=7, do_merge=True)
        lines = fld.detect(img)
    elif method == "hough":
        # Hough变换只能得到直线
        lines = cv2.HoughLines(img, rho=1, theta=np.pi / 2, threshold=100, srn=0, stn=5, min_theta=0, max_theta=1)
        if lines is not None:
            def polar2cartesian(lines):
                for rho, theta in lines[0]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 100 * (-b))
                    y1 = int(y0 + 100 * (a))
                    x2 = int(x0 - 100 * (-b))
                    y2 = int(y0 - 100 * (a))
                return (x0, y0, x2, y2)
            lines = list(map(polar2cartesian, lines))
    elif method == "hough_p":
        # only detect vertical and horizontal lines
        lines = cv2.HoughLinesP(img, rho=1, theta=np.pi / 180, threshold=300, minLineLength=50, maxLineGap=5)
    elif method == "ensemble":
        lsd = cv2.createLineSegmentDetector(refine=cv2.LSD_REFINE_STD, scale=1.0, ang_th=35, quant=3)
        lines_lsd, width, prec, nfa = lsd.detect(img)
        # fld = cv2.ximgproc.createFastLineDetector(distance_threshold=10, canny_th1=30, canny_th2=100, canny_aperture_size=7, do_merge=True)
        # fld = cv2.createFastLineDetector(distance_threshold=10, canny_th1=30, canny_th2=100, canny_aperture_size=7, do_merge=True)
        # lines_fld = fld.detect(img)
        # lines_hough_p = cv2.HoughLinesP(img, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=5)
        lines_hough_p = cv2.HoughLinesP(img, rho=1, theta=np.pi/2, threshold=20, minLineLength=50, maxLineGap=10)

        # 检查检测结果是否为空，并过滤掉空结果
        valid_lines = []
        if lines_lsd is not None and len(lines_lsd) > 0:
            valid_lines.append(lines_lsd)
        # if lines_fld is not None and len(lines_fld) > 0:
        #     valid_lines.append(lines_fld)
        if lines_hough_p is not None and len(lines_hough_p) > 0:
            valid_lines.append(lines_hough_p)

        if valid_lines:
            lines = np.concatenate(valid_lines, axis=0)
        else:
            lines = []

    else:
        print(f"Error: Invalid line detection method '{method}' for image {imgname}!")
        return None

    if lines is None or len(lines) == 0:
        print(f"No lines detected for image {imgname} using method {method}.")
        return []

    line_list1, line_list2 = filter_lines(lines)
    # return line_list1 + line_list2   

    # lines_res1 = merge_lines_by_contour(line_list1, shape)
    # lines_res1 = merge_line(line_list1)
    lines_res1 = line_det_hough(img)    
    lines_res2 = merge_lines_by_overlap(line_list2)

    return lines_res1 + lines_res2


# 水平垂直直线检测-霍夫
def test_pipeline_line_detection_hough():
    """
    测试流水线
    输入：含有圆弧拐角的构件的像素矩阵
    输出：圆弧的矢量化信息、绘制的结果
    """
    log_root = r"/home/chenzhuofan/pipeline/project/modules/test_logs"
    # logname = '直线检测_管道_精智-test-接口测试'   
    # is_visulized = True
    # logname = '直线检测_管道_精智-test-直线检测-无角度过滤_无长度过滤'  
    logname = '直线检测_管道_精智-test-直线检测-低阈值霍夫-20'  
    is_visulized = False

    # 创建文件夹
    os.makedirs(f'{log_root}/{logname}', exist_ok=True)




    # 目标文件夹
    img_dir = r"/home/chenzhuofan/pipeline/project/modules/test_logs/精智构件图-第二次标注-整理"
    # img_dir = r"/home/chenzhuofan/pipeline/project/modules/test_logs/精智构件图-第二次标注-整理-test-接口测试"
    # 获取文件夹下，所有文件的文件名
    filenames = os.listdir(img_dir)
    # 去除标注
    with alive_bar(len(filenames)) as bar:
        for filename in filenames:
            prefix, suffix = os.path.splitext(filename)
            # 读取图像
            image = cv2.imread(os.path.join(img_dir, filename))
            # 对图像进行上下翻转
            image = cv2.flip(image, 0)


            
            # 建立空列表，存储线条，圆形，圆弧的信息
            lines_final = []
            circles_final = []
            arcs_final = []


            

            # 提升两倍分辨率
            # image = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2), interpolation=cv2.INTER_CUBIC)
            image = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2))
            # 保存图像
            if is_visulized:
                cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_1_resize{suffix}"), image)



            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 二值化，且黑白颠倒
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # 保存binary
            if is_visulized:
                cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_2_binary{suffix}"), binary)



            # 腐蚀
            kernel_size = 3
            eroded = cv2.erode(binary, np.ones((kernel_size, kernel_size), np.uint8), iterations=2)
            # 去除细小物体
            eroded = remove_tiny(eroded)
            # 保存eroded
            if is_visulized:
                cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_3-0_eroded-2{suffix}"), eroded)



            # 膨胀
            dilated_1 = cv2.dilate(eroded, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
            # 保存dilated
            if is_visulized:
                cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_3_eroded-2_dilated-1_final-used{suffix}"), dilated_1)

            # # 膨胀
            # dilated_2 = cv2.dilate(eroded, np.ones((kernel_size, kernel_size), np.uint8), iterations=2)
            # # 保存dilated
            # cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_3-0_eroded_dilated-2{suffix}"), dilated_2)

            eroded = dilated_1



            # 去除箭头：将面积在340~410的轮廓，使用黑色填充
            # 查找轮廓
            mode = cv2.RETR_LIST
            contours, _ = cv2.findContours(eroded, mode=mode, method=cv2.CHAIN_APPROX_SIMPLE)
            # 遍历轮廓
            for contour in contours:
                # 计算面积
                area = cv2.contourArea(contour)
                # 两次腐蚀+一次膨胀
                # 将面积在340~410的轮廓，使用黑色填充
                if 220 < area < 300:
                    cv2.fillPoly(eroded, [contour], 0)

                # # 仅两次腐蚀
                # if 100 < area < 200:
                #     cv2.fillPoly(eroded, [contour], 0)
            arrows_removed = eroded
            arrows_removed = remove_tiny(arrows_removed)
            # 保存arrows_fitting_removed
            if is_visulized:
                cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_4_arrows_removed{suffix}"), arrows_removed)




            # 去除外部元素
            # 查找轮廓
            mode = cv2.RETR_EXTERNAL
            contours, _ = cv2.findContours(arrows_removed, mode=mode, method=cv2.CHAIN_APPROX_SIMPLE)
            # 遍历轮廓
            for contour in contours:
                # 计算轮廓的面积
                area = cv2.contourArea(contour)
                if area < 3000:
                    # 对对应的灰度图区域使用黑色填充
                    cv2.fillPoly(arrows_removed, [contour], 0)
            outside_removed = arrows_removed
            # 保存outside_removed
            if is_visulized:
                cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_5_outside_removed{suffix}"), outside_removed)




            # 去除圆形：圆形度筛选
            # 查找轮廓
            mode = cv2.RETR_LIST
            contours, _ = cv2.findContours(outside_removed, mode=mode, method=cv2.CHAIN_APPROX_SIMPLE)
            # 遍历轮廓
            for contour in contours:
                # 计算圆形度
                if abs(cv2.arcLength(contour, True)) < 1e-9:
                    continue
                circularity = 4 * np.pi * cv2.contourArea(contour) / (cv2.arcLength(contour, True) ** 2)
                if circularity > 0.85:
                    # 对对应的灰度图区域使用黑色填充
                    cv2.fillPoly(outside_removed, [contour], 0)
                    # 计算对应圆形和半径
                    center, radius = cv2.minEnclosingCircle(contour)
                    circles_final.append((center[0], center[1], radius))
            circles_removed = outside_removed

            # 检测圆形的双重保险
            # 在一重腐蚀上，再次检测。在outside_removed上，进行填充
            eroded_1 = cv2.erode(binary, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
            mode = cv2.RETR_LIST
            contours, _ = cv2.findContours(eroded_1, mode=mode, method=cv2.CHAIN_APPROX_SIMPLE)
            # 遍历轮廓
            for contour in contours:
                # 计算圆形度
                if abs(cv2.arcLength(contour, True)) < 1e-9:
                    continue
                circularity = 4 * np.pi * cv2.contourArea(contour) / (cv2.arcLength(contour, True) ** 2)
                if circularity > 0.85:
                    # 对对应的灰度图区域使用黑色填充
                    cv2.fillPoly(outside_removed, [contour], 0)
                    # 计算对应圆形和半径
                    center, radius = cv2.minEnclosingCircle(contour)
                    circles_final.append((center[0], center[1], radius))
            # 保存circles_removed
            if is_visulized:
                cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_6_circles_removed{suffix}"), circles_removed)



            # 腐蚀
            kernel_size = 3
            eroded = cv2.erode(circles_removed, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
            # 保存eroded
            if is_visulized:
                cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_6-1_eroded{suffix}"), eroded) 
            circles_removed = eroded


            # 直线检测和直线去除分开进行 
            # 直线检测-霍夫概率
            # lines = cv2.HoughLinesP(circles_removed, rho=1, theta=np.pi/2, threshold=30, minLineLength=50, maxLineGap=10)
            lines = cv2.HoughLinesP(circles_removed, rho=1, theta=np.pi/2, threshold=20, minLineLength=50, maxLineGap=10)
            if lines is not None:
                # temp = image.copy()
                # for line in lines:
                #     x1, y1, x2, y2 = line[0]
                #     # cv2.line(circles_removed, (x1, y1), (x2, y2), (0, 0, 0), 3)
                #     # 在temp上，随机颜色绘制直线
                #     cv2.line(temp, (x1, y1), (x2, y2), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 3)
                # # 保存结果
                # # cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_7-0-0_lines_removed_hough{suffix}"), circles_removed)
                # cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_7-0-1_lines_detect_hough_original{suffix}"), temp)

                # 直线优化，去除相近直线
                line_list = []
                for idx, line in enumerate(lines):
                    x1, y1, x2, y2 = line[0]
                    complicated = False
                    for l in line_list:
                        if math.sqrt((l[0] - x1)**2 + (l[1] - y1)**2 + (l[2] - x2)**2 + (l[3] - y2)**2) <= 3:
                            complicated = True
                            break
                    if not complicated:
                        # line_list.append([int(x1), int(y1), int(x2), int(y2)])
                        line_list.append([x1, y1, x2, y2])
                # # 绘制直线，并保存结果
                # temp = image.copy()
                # for line in line_list:
                #     x1, y1, x2, y2 = line
                #     # 随机颜色绘制直线
                #     cv2.line(temp, (x1, y1), (x2, y2), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 3)
                # # 保存结果
                # cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_7-0-1_lines_detection_hough_optimized-1{suffix}"), temp)

                lines_vertical_horizonal = merge_line(line_list)
                # 将结果添加进lines_final
                # lines_final.extend(lines_vertical_horizonal)

                # 绘制直线
                temp = image.copy()
                for line in lines_vertical_horizonal:
                    x1, y1, x2, y2 = line
                    cv2.line(temp, (x1, y1), (x2, y2), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 5)
                cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_7-0_lines_detection_hough_optimized{suffix}"), temp)
                temp = image.copy()
                for line in lines_vertical_horizonal:
                    x1, y1, x2, y2 = line
                    cv2.line(temp, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_7-0_lines_detection_hough_optimized-red{suffix}"), temp)
                # 保存结果
                # cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_7-0-2_lines_detection_hough_optimized-2{suffix}"), temp)

            else:
                print(f"No lines detected for image {filename} using method hough.")




            bar()


# 完整直线检测-师清
def test_pipeline_line_detection():
    """
    测试流水线
    输入：含有圆弧拐角的构件的像素矩阵
    输出：圆弧的矢量化信息、绘制的结果
    """
    log_root = r"/home/chenzhuofan/pipeline/project/modules/test_logs"
    # logname = '直线检测_管道_精智-test-接口测试'   
    # is_visulized = True
    # logname = '直线检测_管道_精智-test-直线检测-无角度过滤_无长度过滤'  
    logname = '直线检测_管道_精智-test-直线检测-低阈值霍夫'  
    is_visulized = False

    # 创建文件夹
    os.makedirs(f'{log_root}/{logname}', exist_ok=True)




    # 目标文件夹
    img_dir = r"/home/chenzhuofan/pipeline/project/modules/test_logs/精智构件图-第二次标注-整理"
    # img_dir = r"/home/chenzhuofan/pipeline/project/modules/test_logs/精智构件图-第二次标注-整理-test-接口测试"
    # 获取文件夹下，所有文件的文件名
    filenames = os.listdir(img_dir)
    # 去除标注
    with alive_bar(len(filenames)) as bar:
        for filename in filenames:
            prefix, suffix = os.path.splitext(filename)
            # 读取图像
            image = cv2.imread(os.path.join(img_dir, filename))
            # 对图像进行上下翻转
            image = cv2.flip(image, 0)


            
            # 建立空列表，存储线条，圆形，圆弧的信息
            lines_final = []
            circles_final = []
            arcs_final = []


            

            # 提升两倍分辨率
            # image = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2), interpolation=cv2.INTER_CUBIC)
            image = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2))
            # 保存图像
            if is_visulized:
                cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_1_resize{suffix}"), image)



            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 二值化，且黑白颠倒
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # 保存binary
            if is_visulized:
                cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_2_binary{suffix}"), binary)



            # 腐蚀
            kernel_size = 3
            eroded = cv2.erode(binary, np.ones((kernel_size, kernel_size), np.uint8), iterations=2)
            # 去除细小物体
            eroded = remove_tiny(eroded)
            # 保存eroded
            if is_visulized:
                cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_3-0_eroded-2{suffix}"), eroded)



            # 膨胀
            dilated_1 = cv2.dilate(eroded, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
            # 保存dilated
            if is_visulized:
                cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_3_eroded-2_dilated-1_final-used{suffix}"), dilated_1)

            # # 膨胀
            # dilated_2 = cv2.dilate(eroded, np.ones((kernel_size, kernel_size), np.uint8), iterations=2)
            # # 保存dilated
            # cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_3-0_eroded_dilated-2{suffix}"), dilated_2)

            eroded = dilated_1


            # 去除箭头：将面积在340~410的轮廓，使用黑色填充
            # 查找轮廓
            mode = cv2.RETR_LIST
            contours, _ = cv2.findContours(eroded, mode=mode, method=cv2.CHAIN_APPROX_SIMPLE)
            # 遍历轮廓
            for contour in contours:
                # 计算面积
                area = cv2.contourArea(contour)
                # 两次腐蚀+一次膨胀
                # 将面积在340~410的轮廓，使用黑色填充
                if 220 < area < 300:
                    cv2.fillPoly(eroded, [contour], 0)

                # # 仅两次腐蚀
                # if 100 < area < 200:
                #     cv2.fillPoly(eroded, [contour], 0)
            arrows_removed = eroded
            arrows_removed = remove_tiny(arrows_removed)
            # 保存arrows_fitting_removed
            if is_visulized:
                cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_4_arrows_removed{suffix}"), arrows_removed)




            # 去除外部元素
            # 查找轮廓
            mode = cv2.RETR_EXTERNAL
            contours, _ = cv2.findContours(arrows_removed, mode=mode, method=cv2.CHAIN_APPROX_SIMPLE)
            # 遍历轮廓
            for contour in contours:
                # 计算轮廓的面积
                area = cv2.contourArea(contour)
                if area < 3000:
                    # 对对应的灰度图区域使用黑色填充
                    cv2.fillPoly(arrows_removed, [contour], 0)
            outside_removed = arrows_removed
            # 保存outside_removed
            if is_visulized:
                cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_5_outside_removed{suffix}"), outside_removed)




            # 去除圆形：圆形度筛选
            # 查找轮廓
            mode = cv2.RETR_LIST
            contours, _ = cv2.findContours(outside_removed, mode=mode, method=cv2.CHAIN_APPROX_SIMPLE)
            # 遍历轮廓
            for contour in contours:
                # 计算圆形度
                if abs(cv2.arcLength(contour, True)) < 1e-9:
                    continue
                circularity = 4 * np.pi * cv2.contourArea(contour) / (cv2.arcLength(contour, True) ** 2)
                if circularity > 0.85:
                    # 对对应的灰度图区域使用黑色填充
                    cv2.fillPoly(outside_removed, [contour], 0)
                    # 计算对应圆形和半径
                    center, radius = cv2.minEnclosingCircle(contour)
                    circles_final.append((center[0], center[1], radius))
            circles_removed = outside_removed

            # 检测圆形的双重保险
            # 在一重腐蚀上，再次检测。在outside_removed上，进行填充
            eroded_1 = cv2.erode(binary, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
            mode = cv2.RETR_LIST
            contours, _ = cv2.findContours(eroded_1, mode=mode, method=cv2.CHAIN_APPROX_SIMPLE)
            # 遍历轮廓
            for contour in contours:
                # 计算圆形度
                if abs(cv2.arcLength(contour, True)) < 1e-9:
                    continue
                circularity = 4 * np.pi * cv2.contourArea(contour) / (cv2.arcLength(contour, True) ** 2)
                if circularity > 0.85:
                    # 对对应的灰度图区域使用黑色填充
                    cv2.fillPoly(outside_removed, [contour], 0)
                    # 计算对应圆形和半径
                    center, radius = cv2.minEnclosingCircle(contour)
                    circles_final.append((center[0], center[1], radius))
            # 保存circles_removed
            if is_visulized:
                cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_6_circles_removed{suffix}"), circles_removed)



            # 腐蚀
            kernel_size = 3
            eroded = cv2.erode(circles_removed, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
            # 保存eroded
            if is_visulized:
                cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_6-1_eroded{suffix}"), eroded) 
            circles_removed = eroded



            # 直线检测-师清版
            lines = line_det(circles_removed, image.shape, filename)
            # 随机颜色绘制直线
            temp = image.copy()
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line
                    cv2.line(temp, (x1, y1), (x2, y2), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 5)
            # 保存结果
            cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_7-0_lines_detection-shiqing{suffix}"), temp)
            # 红色绘制直线
            temp = image.copy()
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line
                    cv2.line(temp, (x1, y1), (x2, y2), (0, 0, 255), 5)
            # 保存结果
            cv2.imwrite(os.path.join(log_root, logname, f"{prefix}_7-0_lines_detection-red-shiqing{suffix}"), temp)


            bar()


# 构件直线检测
def component_line_det(img_path_list):
    """
    输入图像地址列表，调用line_det函数遍历检测图像直线，建立字典res存储检测结果，key为地址，value为检测结果
    """
    res = {}

    for img_path in img_path_list:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to load image {img_path}")
            continue

        img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel_size = 3
        eroded = cv2.erode(binary, np.ones((kernel_size, kernel_size), np.uint8), iterations=2)
        eroded = remove_tiny(eroded)

        dilated_1 = cv2.dilate(eroded, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
        eroded = dilated_1

        mode = cv2.RETR_LIST
        contours, _ = cv2.findContours(eroded, mode=mode, method=cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            # 两次腐蚀+一次膨胀
            # 将面积在340~410的轮廓，使用黑色填充
            if 220 < area < 300:
                cv2.fillPoly(eroded, [contour], 0)
            # # 仅两次腐蚀
            # if 100 < area < 200:
            #     cv2.fillPoly(eroded, [contour], 0)
        arrows_removed = eroded
        arrows_removed = remove_tiny(arrows_removed)

        mode = cv2.RETR_EXTERNAL
        contours, _ = cv2.findContours(arrows_removed, mode=mode, method=cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 3000:
                cv2.fillPoly(arrows_removed, [contour], 0)
        outside_removed = arrows_removed


        mode = cv2.RETR_LIST
        contours, _ = cv2.findContours(outside_removed, mode=mode, method=cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if abs(cv2.arcLength(contour, True)) < 1e-9:
                continue
            circularity = 4 * np.pi * cv2.contourArea(contour) / (cv2.arcLength(contour, True) ** 2)
            if circularity > 0.85:
                cv2.fillPoly(outside_removed, [contour], 0)
        
        eroded_1 = cv2.erode(binary, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
        mode = cv2.RETR_LIST
        contours, _ = cv2.findContours(eroded_1, mode=mode, method=cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if abs(cv2.arcLength(contour, True)) < 1e-9:
                continue
            circularity = 4 * np.pi * cv2.contourArea(contour) / (cv2.arcLength(contour, True) ** 2)
            if circularity > 0.85:
                cv2.fillPoly(outside_removed, [contour], 0)
        circles_removed = outside_removed



        kernel_size = 3
        eroded = cv2.erode(circles_removed, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
        circles_removed = eroded


        shape = img.shape
        lines_detected = line_det_component(circles_removed, shape, img_path)
        lines_detected = [[x / 2 for x in sublist] for sublist in lines_detected]
        res[img_path] = lines_detected

    return res
# 代码测试
def test_line_det():
    log_root = '/home/chenzhuofan/project_que/pipeline_jingzhi/logs'
    logname = '构件直线检测demo展示'

    img_dir = '/home/chenzhuofan/project_que/pipeline_jingzhi/data/精智demo展示案例备选二'
    img_path_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
    if not os.path.exists(os.path.join(log_root, logname)):
        os.makedirs(os.path.join(log_root, logname))

    res = annotation_line_det(img_path_list)
    # print(res)

    for img_path, lines in res.items():
        baseneame = os.path.basename(img_path)
        prefix, suffix = os.path.splitext(baseneame)
        image = cv2.imread(img_path)
        image_red = image.copy()
        for line in lines:
            x1, y1, x2, y2 = line
            # 随机颜色绘制
            color = [random.randint(0, 255) for _ in range(3)]
            cv2.line(image, (x1, y1), (x2, y2), color, 2)
            # 红色绘制
            cv2.line(image_red, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # 保存图片
        output_path = os.path.join(log_root, logname, prefix + '_line_det' + suffix)
        cv2.imwrite(output_path, image)
        output_path_red = os.path.join(log_root, logname, prefix + '_line_det_red' + suffix)
        cv2.imwrite(output_path_red, image_red)
        # print('output_path:', output_path)
        # # 展示图片
        # cv2.imshow('image', image)
        # cv2.imshow('image_red', image_red)
        # cv2.waitKey(0)



# 标注直线检测（所有直线检测）
def annotation_line_det(img_path_list):
    """
    输入图像地址列表，调用line_det函数遍历检测图像直线，建立字典res存储检测结果，key为地址，value为检测结果
    """
    res = {}

    for img_path in img_path_list:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to load image {img_path}")
            continue

        # 图像预处理：灰度化，二值化且黑白颠倒
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        shape = img.shape
        lines_detected = line_det(binary, shape, img_path)
        res[img_path] = lines_detected

    return res
# 代码测试
def test_annotation_line_det():
    log_root = '/home/chenzhuofan/project_que/pipeline_jingzhi/logs'
    logname = '标注直线检测demo展示'

    img_dir = '/home/chenzhuofan/project_que/pipeline_jingzhi/data/精智demo展示案例备选二'
    img_path_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
    if not os.path.exists(os.path.join(log_root, logname)):
        os.makedirs(os.path.join(log_root, logname))

    res = annotation_line_det(img_path_list)
    # print(res)

    for img_path, lines in res.items():
        baseneame = os.path.basename(img_path)
        prefix, suffix = os.path.splitext(baseneame)
        image = cv2.imread(img_path)
        image_red = image.copy()
        for line in lines:
            x1, y1, x2, y2 = line
            # 随机颜色绘制
            color = [random.randint(0, 255) for _ in range(3)]
            cv2.line(image, (x1, y1), (x2, y2), color, 2)
            # 红色绘制
            cv2.line(image_red, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # 保存图片
        output_path = os.path.join(log_root, logname, prefix + '_line_det' + suffix)
        cv2.imwrite(output_path, image)
        output_path_red = os.path.join(log_root, logname, prefix + '_line_det_red' + suffix)
        cv2.imwrite(output_path_red, image_red)
        # print('output_path:', output_path)
        # # 展示图片
        # cv2.imshow('image', image)
        # cv2.imshow('image_red', image_red)
        # cv2.waitKey(0)


if __name__ == "__main__":
    test_process_image()

    None