import cv2
import numpy as np

def process_bbox(bbox_list, image):
    points_list = []
    for bbox in bbox_list:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        aspect_ratio = w / h
        
        # 提取bbox区域
        bbox_region = image[y1:y2, x1:x2]
        
        # 判断bbox的方向
        if aspect_ratio > 1.2:
            # 二值化处理
            gray = cv2.cvtColor(bbox_region, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # 左右均分
            mid_x = w // 2
            left_region = binary[:, :mid_x]
            right_region = binary[:, mid_x:]
            
            # 计算黑色像素（0值）的数量
            left_black_pixels = np.sum(left_region == 0)
            right_black_pixels = np.sum(right_region == 0)
            
            # 判断黑色像素较少的一侧并取中间高度的点
            mid_y = y1 + h // 2
            if left_black_pixels < right_black_pixels:
                point = (x1, mid_y)
            else:
                point = (x2, mid_y)
            
            points_list.append(point)
        
        elif aspect_ratio < 0.8:
            # 二值化处理
            gray = cv2.cvtColor(bbox_region, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # 上下均分
            mid_y = h // 2
            top_region = binary[:mid_y, :]
            bottom_region = binary[mid_y:, :]
            
            # 计算黑色像素（0值）的数量
            top_black_pixels = np.sum(top_region == 0)
            bottom_black_pixels = np.sum(bottom_region == 0)
            
            # 判断黑色像素较少的一侧并取中间宽度的点
            mid_x = x1 + w // 2
            if top_black_pixels < bottom_black_pixels:
                point = (mid_x, y1)
            else:
                point = (mid_x, y2)
            
            points_list.append(point)

    return points_list

def get_distance_point2line(point, line):
    line_point1, line_point2 = np.array(line[0:2]), np.array(line[2:])
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
    return distance

def project_point_onto_line(point, line):
    x1, y1, x2, y2 = line
    px, py = point
    line_vec = np.array([x2 - x1, y2 - y1])
    point_vec = np.array([px - x1, py - y1])
    line_len_sq = np.dot(line_vec, line_vec)
    t = np.dot(point_vec, line_vec) / line_len_sq
    qx = x1 + t * line_vec[0]
    qy = y1 + t * line_vec[1]
    return [qx, qy]

def project_in_line(project, line, threshold):
    px, py = project
    x1, y1, x2, y2 = line
    if x1 - threshold < px < x2 + threshold and y1 -threshold < py < y2 + threshold:
        return True
    else:
        return False

def split_lines_by_points(lines, points):
    split_lines = []
    threshold = 5  # 距离阈值
    
    for line in lines:
        relevant_points = []
        i = 0
        for point in points:
            project = project_point_onto_line(point, line)
            if get_distance_point2line(point, line) < threshold and project_in_line(project, line, 10):
                relevant_points.append(project)
            i = i + 1
        if len(relevant_points) > 1:
            relevant_points.sort(key=lambda p: (p[0], p[1]))
            for i in range(len(relevant_points) - 1):
                x1, y1 = relevant_points[i]
                x2, y2 = relevant_points[i + 1]
                split_lines.append((int(x1), int(y1), int(x2), int(y2)))
    return split_lines

def pipeline_split_line(res_lines, arrows):
    split_lines = {}
    for img_path, lines in res_lines.items():
        image = cv2.imread(img_path)
        bbox_list = arrows[img_path]
        points = process_bbox(bbox_list, image)
        split_lines[img_path] = split_lines_by_points(lines, points)
    return split_lines



import sys
import arrow_det_modules
from line_det_module import annotation_line_det, component_line_det
import cv2
import random
import os


log_root = './logs'
logname = '构件直线分割demo展示_精智'

img_dir = './data/精智demo展示案例备选二'
img_path_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
if not os.path.exists(os.path.join(log_root, logname)):
    os.makedirs(os.path.join(log_root, logname))

res_line = annotation_line_det(img_path_list)
arrow_dic = arrow_det_modules.pipeline_arrow_det(img_path_list)
split_lines = pipeline_split_line(res_line, arrow_dic)

for img_path, split_line in split_lines.items():
    baseneame = os.path.basename(img_path)
    prefix, suffix = os.path.splitext(baseneame)
    image = cv2.imread(img_path)
    for line in split_line:
        x1, y1, x2, y2 = line
        # 随机颜色绘制
        color = [random.randint(0, 255) for _ in range(3)]
        cv2.line(image, (x1, y1), (x2, y2), color, 2)
    output_path = os.path.join(log_root, logname, prefix + '_line_split' + suffix)
    cv2.imwrite(output_path, image)
