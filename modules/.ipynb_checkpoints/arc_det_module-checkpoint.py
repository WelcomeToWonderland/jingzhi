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


def remove_tiny(image, area_threshold=50):
    """
    输入：背景为黑，目标为白的二值图像
    处理：将其中面积小于指定阈值的细小物体使用黑色覆盖
    输出：处理后的二值图像

    参数:
    - image: 输入的二值图像
    - area_threshold: 面积阈值，低于此值的轮廓将被填充为黑色（默认值为50）
    """
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        # 将面积小于指定阈值的轮廓填充为黑色
        if area < area_threshold:
            cv2.fillPoly(image, [contour], 0)
    return image

# def generate_graph_from_component(component):
#     # todo: 点坐标与像素矩阵下标是相反的，需要注意
#     # 假设component是一个二值图像，代表单个连通分量
#     G = nx.Graph()
#     height, width = component.shape
#     for y in range(height):
#         for x in range(width):
#             if component[y, x] == 255:  # 假定白色像素是连通分量的一部分
#                 G.add_node((y, x))  # 添加节点
#                 # 八个方向的像素
#                 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
#                     yy, xx = y + dy, x + dxk
#                     if 0 <= yy < height and 0 <= xx < width and component[yy, xx] == 255:
#                         G.add_edge((y, x), (yy, xx), weight=1)  # 假设边的权重都是1
#     return G

def generate_graph_from_component(component):
    # todo: 点坐标与像素矩阵下标是相反的，需要注意
    # 假设component是一个二值图像，代表单个连通分量
    G = nx.Graph()
    height, width = component.shape
    for y in range(height):
        for x in range(width):
            if component[y, x] == 255:  # 假定白色像素是连通分量的一部分
                G.add_node((x, y))  # 添加节点，使用（横坐标，纵坐标）
                # 八个方向的像素
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < width and 0 <= yy < height and component[yy, xx] == 255:
                        G.add_edge((x, y), (xx, yy), weight=1)  # 假设边的权重都是1
    return G


# todo: 更新斜率计算算法，避免出现除0错误
# def find_intersection(point_a, point_b, point_c):
    
#     def calculate_midpoint(point1, point2):
#         """计算两点的中点"""
#         return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

#     def line_slope(point1, point2):
#         """计算两点间直线的斜率，垂直线返回np.inf"""
#         dx = point2[0] - point1[0]
#         if dx == 0:
#             return np.inf
#         return (point2[1] - point1[1]) / dx

#     def perpendicular_slope(slope):
#         """计算给定斜率的垂直线斜率，水平线返回np.inf"""
#         if slope == 0:  # 处理原线为水平线的情况
#             return np.inf
#         return -1 / slope


#     """
#     计算由三个点构成的两对点的中垂线的交点，并返回这些交点的平均值。
#     """
#     # 将三个点的数据类型转换为浮点数
#     point_a = np.array(point_a, dtype=np.float64)
#     point_b = np.array(point_b, dtype=np.float64)
#     point_c = np.array(point_c, dtype=np.float64)

#     mid_ab = calculate_midpoint(point_a, point_b)
#     mid_bc = calculate_midpoint(point_b, point_c)
    
#     slope_ab = line_slope(point_a, point_b)
#     slope_bc = line_slope(point_b, point_c)
    
#     perp_slope_ab = perpendicular_slope(slope_ab)
#     perp_slope_bc = perpendicular_slope(slope_bc)
    
#     # 计算两中垂线的交点
#     if perp_slope_ab == np.inf:  # AB的中垂线为垂直线
#         x_intersect = mid_ab[0]
#         y_intersect = perp_slope_bc * (x_intersect - mid_bc[0]) + mid_bc[1]
#     elif perp_slope_bc == np.inf:  # BC的中垂线为垂直线
#         x_intersect = mid_bc[0]
#         y_intersect = perp_slope_ab * (x_intersect - mid_ab[0]) + mid_ab[1]
#     else:  # 两中垂线都不是垂直线
#         # 解线性方程组求交点
#         A = np.array([[-perp_slope_ab, 1], [-perp_slope_bc, 1]])
#         b = np.array([mid_ab[1] - perp_slope_ab * mid_ab[0], mid_bc[1] - perp_slope_bc * mid_bc[0]])
#         intersection = np.linalg.solve(A, b)
#         x_intersect, y_intersect = intersection
    
#     # 返回交点的平均值（本例中只有一个交点，其平均即为其自身）
#     return (x_intersect, y_intersect)


# def arc_det(img):
#     """
#     输入完成预处理的构件图像
#     """
#     # 提取连通分量
#     num_labels, labeled_image = cv2.connectedComponents(img)
#     for label in range(1, num_labels):
#         component = np.uint8(labeled_image == label) * 255
#         G = generate_graph_from_component(component)
#         # 检测component的轮廓
#         contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         # 计算所有点对的最短路径长度
#         try:
#             all_pairs_dists = dict(nx.all_pairs_shortest_path_length(G))
#             # 找到最长距离及其对应的两点
#             max_dist = 0
#             farthest_nodes = None
#             for node1, dists in all_pairs_dists.items():
#                 for node2, dist in dists.items():
#                     if dist > max_dist:
#                         max_dist = dist
#                         farthest_nodes = (node1, node2)

#             if farthest_nodes is not None:
#                 # 计算中垂线（确保考虑各种方向）
#                 y1, x1 = farthest_nodes[0]
#                 y2, x2 = farthest_nodes[1]
#                 dx = x2 - x1
#                 dy = y2 - y1
#                 mid_point = ((x1 + x2) // 2, (y1 + y2) // 2)
                
#                 # 中垂线的计算
#                 if dx == 0:  # 原两点连线垂直
#                     midline_slope = 0  # 中垂线水平
#                     midline_intercept = mid_point[1]  # 中点y坐标作为截距
#                 elif dy == 0:  # 原两点连线水平
#                     midline_slope = None  # 中垂线垂直，无需斜率
#                     midline_intercept = mid_point[0]  # 中点x坐标作为截距位置
#                 else:  # 一般情况
#                     midline_slope = -dx / dy  # 中垂线斜率
#                     midline_intercept = mid_point[1] - midline_slope * mid_point[0]  # 计算截距
                
#                 # 找到中垂线与轮廓的交点
#                 intersection_points = []
#                 for contour in contours:
#                     for point in contour.squeeze():
#                         x, y = point
#                         # 计算交点逻辑
#                         if midline_slope is None:  # 垂直中垂线
#                             if abs(x - midline_intercept) < 10:  # 近似判断
#                                 intersection_points.append(point)
#                         elif midline_slope == 0:  # 水平中垂线
#                             if abs(y - midline_intercept) < 10:  # 近似判断
#                                 intersection_points.append(point)
#                         else:  # 斜线中垂线
#                             if abs(y - (midline_slope * x + midline_intercept)) < 10:  # 近似判断
#                                 intersection_points.append(point)


#                 # 计算交点的平均值
#                 if len(intersection_points) > 0:
#                     avg_x = sum(pt[0] for pt in intersection_points) / len(intersection_points)
#                     avg_y = sum(pt[1] for pt in intersection_points) / len(intersection_points)

#                     # 将farthest_nodes中坐标颠倒
#                     farthest_nodes = [(x, y) for y, x in farthest_nodes]
#                     center = find_intersection(farthest_nodes[0], (avg_x, avg_y), farthest_nodes[1])
#                     # 求出center与farthest_nodes和(avg_x, avg_y)三个点的距离平均值
#                     radius = (np.linalg.norm(np.array(center) - np.array(farthest_nodes[0])) + np.linalg.norm(np.array(center) - np.array(farthest_nodes[1])) + np.linalg.norm(np.array(center) - np.array((avg_x, avg_y)))) / 3
                    
#                     # 将结果按照center_x, center_y, radius, 0, 0的形式添加进arcs_final中
#                     # arcs_final.append((center[0], center[1], radius, 0, 0))
#                 else:
#                     print("中垂线与轮廓的交点检测失败") 
#             else:
#                 print("圆弧端点检测失败")
#                 # # 绘制轮廓
#                 # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
#                 # # 在轮廓附近标注“圆弧端点检测失败”
#                 # cv2.putText(image, "Arc endpoints detection failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
#         except nx.NetworkXError:
#             # 处理空图或只有一个节点的情况
#             pass



# 函数更新
def find_perpendicular_bisector_arc_intersection(line, contours):
    """
    计算给定直线的中垂线，并找到中垂线与轮廓的交点，返回交点的平均值
    """
    (x1, y1), (x2, y2) = line
    dx = x2 - x1
    dy = y2 - y1
    mid_point = ((x1 + x2) // 2, (y1 + y2) // 2)
    
    # 中垂线的计算
    if dx == 0:  # 原两点连线垂直
        midline_slope = 0  # 中垂线水平
        midline_intercept = mid_point[1]  # 中点 y 坐标作为截距
    elif dy == 0:  # 原两点连线水平
        midline_slope = None  # 中垂线垂直，无需斜率
        midline_intercept = mid_point[0]  # 中点 x 坐标作为截距位置
    else:  # 一般情况
        midline_slope = -dx / dy  # 中垂线斜率
        midline_intercept = mid_point[1] - midline_slope * mid_point[0]  # 计算截距
    
    # 找到中垂线与轮廓的交点
    intersection_points = []
    for contour in contours:
        for point in contour.squeeze():
            x, y = point
            # 计算交点逻辑
            if midline_slope is None:  # 垂直中垂线
                if abs(x - midline_intercept) < 10:  # 近似判断
                    intersection_points.append(point)
            elif midline_slope == 0:  # 水平中垂线
                if abs(y - midline_intercept) < 10:  # 近似判断
                    intersection_points.append(point)
            else:  # 斜线中垂线
                if abs(y - (midline_slope * x + midline_intercept)) < 10:  # 近似判断
                    intersection_points.append(point)
    
    if len(intersection_points) > 0:
        avg_x = sum(pt[0] for pt in intersection_points) / len(intersection_points)
        avg_y = sum(pt[1] for pt in intersection_points) / len(intersection_points)
        return avg_x, avg_y
    else:
        return None

def find_perpendicular_bisector_intersection(line1, line2):
    def calculate_midpoint(point1, point2):
        """计算两点的中点"""
        return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

    def line_slope(point1, point2):
        """计算两点间直线的斜率，避免除以零的情况"""
        dx = point2[0] - point1[0]
        return (point2[1] - point1[1]) / (dx + 1e-7)

    def perpendicular_slope(slope):
        """计算给定斜率的垂直线斜率，水平线返回np.inf"""
        if slope == 0:  # 处理原线为水平线的情况
            return np.inf
        return -1 / slope

    point_a, point_b = line1
    point_c, point_d = line2

    mid_ab = calculate_midpoint(point_a, point_b)
    mid_cd = calculate_midpoint(point_c, point_d)
    
    slope_ab = line_slope(point_a, point_b)
    slope_cd = line_slope(point_c, point_d)
    
    perp_slope_ab = perpendicular_slope(slope_ab)
    perp_slope_cd = perpendicular_slope(slope_cd)
    
    # 计算两中垂线的交点
    if perp_slope_ab == np.inf:  # AB的中垂线为垂直线
        x_intersect = mid_ab[0]
        y_intersect = perp_slope_cd * (x_intersect - mid_cd[0]) + mid_cd[1]
    elif perp_slope_cd == np.inf:  # CD的中垂线为垂直线
        x_intersect = mid_cd[0]
        y_intersect = perp_slope_ab * (x_intersect - mid_ab[0]) + mid_ab[1]
    else:  # 两中垂线都不是垂直线
        # 解线性方程组求交点
        A = np.array([[-perp_slope_ab, 1], [-perp_slope_cd, 1]])
        b = np.array([mid_ab[1] - perp_slope_ab * mid_ab[0], mid_cd[1] - perp_slope_cd * mid_cd[0]])
        intersection = np.linalg.solve(A, b)
        x_intersect, y_intersect = intersection
    
    # 返回交点的平均值（本例中只有一个交点，其平均即为其自身）
    return (x_intersect, y_intersect)

def polar_angle(center, point):
    dx = point[0] - center[0]
    # todo：逻辑存在问题，没有理清
    # dy = -1 * (point[1] - center[1])
    dy = point[1] - center[1]
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360
    return angle

# 阙祥辉版
# def find_arc_angles(component, center, radius, farthest_nodes):
#     angle1 = polar_angle(center, farthest_nodes[0])
#     angle2 = polar_angle(center, farthest_nodes[1])

#     overlap1 = np.zeros_like(component)
#     overlap2 = np.zeros_like(component)

#     cv2.ellipse(overlap1, (int(center[0]), int(center[1])), (int(radius), int(radius)), 0, angle1, angle2, 255, 3)
#     cv2.ellipse(overlap2, (int(center[0]), int(center[1])), (int(radius), int(radius)), 0, angle2, angle1, 255, 3)

#     overlap1 = cv2.bitwise_and(overlap1, component)
#     overlap2 = cv2.bitwise_and(overlap2, component)

#     if np.sum(overlap1) > np.sum(overlap2):
#         return angle1, angle2
#     else:
#         return angle2, angle1

# 师清版
def find_arc_angles(component, center, radius, farthest_nodes):
    angle1 = polar_angle(center, farthest_nodes[0])
    angle2 = polar_angle(center, farthest_nodes[1])

    if angle1 > angle2:
        angle1, angle2 = angle2, angle1

    # show_img(component, "component")

    overlap1 = np.zeros_like(component)
    overlap2 = np.zeros_like(component)

    cv2.ellipse(overlap1, (int(center[0]), int(center[1])), (int(radius), int(radius)), 0, angle1, angle2, 255, 3)
    # show_img(overlap1, "overlap1")
    cv2.ellipse(overlap2, (int(center[0]), int(center[1])), (int(radius), int(radius)), 0, angle2, 360+angle1, 255, 3)
    # show_img(overlap2, "overlap2")

    overlap1 = cv2.bitwise_and(overlap1, component)
    overlap2 = cv2.bitwise_and(overlap2, component)


    if np.sum(overlap1) > np.sum(overlap2):
        return angle1, angle2
    else:
        return angle2, angle1+360

# def arc_det(img):
#     """
#     输入完成预处理的构件图像
#     """
#     # 提取连通分量
#     num_labels, labeled_image = cv2.connectedComponents(img)
    
#     # 如果只有一个连通分量，直接返回 None
#     if num_labels <= 1:
#         return None

#     for label in range(1, num_labels):
#         component = np.uint8(labeled_image == label) * 255
#         G = generate_graph_from_component(component)
#         # 检测 component 的轮廓
#         contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         # 计算所有点对的最短路径长度
#         try:
#             all_pairs_dists = dict(nx.all_pairs_shortest_path_length(G))
#             # 找到最长距离及其对应的两点
#             max_dist = 0
#             farthest_nodes = None
#             for node1, dists in all_pairs_dists.items():
#                 for node2, dist in dists.items():
#                     if dist > max_dist:
#                         max_dist = dist
#                         farthest_nodes = (node1, node2)

#             if farthest_nodes is not None:
#                 line = ((farthest_nodes[0][1], farthest_nodes[0][0]), (farthest_nodes[1][1], farthest_nodes[1][0]))
#                 avg_point = find_perpendicular_bisector_arc_intersection(line, contours)

#                 if avg_point is not None:
#                     avg_x, avg_y = avg_point
#                     line1 = (farthest_nodes[0], (avg_x, avg_y))
#                     line2 = (farthest_nodes[1], (avg_x, avg_y))
#                     center = find_perpendicular_bisector_intersection(line1, line2)
#                     # 求出 center 与 farthest_nodes 和 (avg_x, avg_y) 三个点的距离平均值
#                     radius = (np.linalg.norm(np.array(center) - np.array(farthest_nodes[0])) + 
#                               np.linalg.norm(np.array(center) - np.array(farthest_nodes[1])) + 
#                               np.linalg.norm(np.array(center) - np.array((avg_x, avg_y)))) / 3
                    
#                     # 将结果按照 center_x, center_y, radius, 0, 0 的形式添加进 arcs_final 中
#                     # arcs_final.append((center[0], center[1], radius, 0, 0))
#                 else:
#                     print("中垂线与轮廓的交点检测失败")
#             else:
#                 print("圆弧端点检测失败")
#                 # 绘制轮廓
#                 # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
#                 # 在轮廓附近标注“圆弧端点检测失败”
#                 # cv2.putText(image, "Arc endpoints detection failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
#         except nx.NetworkXError:
#             # 处理空图或只有一个节点的情况
#             pass

# def arc_det(img):
#     """
#     输入完成预处理的构件图像，并返回检测到的圆弧信息和相关点
#     """
#     # 提取连通分量
#     num_labels, labeled_image = cv2.connectedComponents(img)
    
#     # 如果只有一个连通分量，直接返回 None
#     if num_labels <= 1:
#         return None, None

#     # 存储检测到的结果
#     detected_arcs = []
#     pts = []

#     for label in range(1, num_labels):
#         component = np.uint8(labeled_image == label) * 255
#         G = generate_graph_from_component(component)
#         # 检测 component 的轮廓
#         contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         # 计算所有点对的最短路径长度
#         try:
#             all_pairs_dists = dict(nx.all_pairs_shortest_path_length(G))
#             # 找到最长距离及其对应的两点
#             max_dist = 0
#             farthest_nodes = None
#             for node1, dists in all_pairs_dists.items():
#                 for node2, dist in dists.items():
#                     if dist > max_dist:
#                         max_dist = dist
#                         farthest_nodes = (node1, node2)

#             if farthest_nodes is not None:
#                 # 进行了点坐标与像素矩阵下标的转换
#                 line = ((farthest_nodes[0][1], farthest_nodes[0][0]), (farthest_nodes[1][1], farthest_nodes[1][0]))
#                 avg_point = find_perpendicular_bisector_arc_intersection(line, contours)

#                 if avg_point is not None:
#                     avg_x, avg_y = avg_point
#                     line1 = (farthest_nodes[0], (avg_x, avg_y))
#                     line2 = (farthest_nodes[1], (avg_x, avg_y))
#                     center = find_perpendicular_bisector_intersection(line1, line2)
#                     # 求出 center 与 farthest_nodes 和 (avg_x, avg_y) 三个点的距离平均值
#                     radius = (np.linalg.norm(np.array(center) - np.array(farthest_nodes[0])) + 
#                               np.linalg.norm(np.array(center) - np.array(farthest_nodes[1])) + 
#                               np.linalg.norm(np.array(center) - np.array((avg_x, avg_y)))) / 3
                    
#                     # 将结果按照 center_x, center_y, radius, 0, 0 的形式添加进 detected_arcs 中
#                     detected_arcs.append([center[0], center[1], radius, 0, 0])
#                     # 将 farthest_nodes, avg_point 和 center 添加进 pts 中
#                     pts.append((farthest_nodes[0], farthest_nodes[1], (avg_x, avg_y), center))
#                 else:
#                     print("中垂线与轮廓的交点检测失败")
#             else:
#                 print("圆弧端点检测失败")
#                 # 绘制轮廓
#                 # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
#                 # 在轮廓附近标注“圆弧端点检测失败”
#                 # cv2.putText(image, "Arc endpoints detection failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
#         except nx.NetworkXError:
#             # 处理空图或只有一个节点的情况
#             pass
    
#     return detected_arcs, pts

# def arc_det(img):
#     """
#     输入完成预处理的构件图像，并返回检测到的圆弧信息
#     """
#     # 提取连通分量
#     num_labels, labeled_image = cv2.connectedComponents(img)
    
#     # 如果只有一个连通分量，直接返回 None
#     if num_labels <= 1:
#         return None, None

#     # 存储检测到的结果
#     detected_arcs = []
#     pts = []

#     for label in range(1, num_labels):
#         component = np.uint8(labeled_image == label) * 255
#         G = generate_graph_from_component(component)
#         # 检测 component 的轮廓
#         contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         # 计算所有点对的最短路径长度
#         try:
#             all_pairs_dists = dict(nx.all_pairs_shortest_path_length(G))
#             # 找到最长距离及其对应的两点
#             max_dist = 0
#             farthest_nodes = None
#             for node1, dists in all_pairs_dists.items():
#                 for node2, dist in dists.items():
#                     if dist > max_dist:
#                         max_dist = dist
#                         farthest_nodes = (node1, node2)

#             if farthest_nodes is not None:
#                 # 进行点坐标与像素矩阵下标的转换
#                 line = (farthest_nodes[0], farthest_nodes[1])
#                 avg_point = find_perpendicular_bisector_arc_intersection(line, contours)

#                 if avg_point is not None:
#                     avg_x, avg_y = avg_point
#                     line1 = (farthest_nodes[0], (avg_x, avg_y))
#                     line2 = (farthest_nodes[1], (avg_x, avg_y))
#                     center = find_perpendicular_bisector_intersection(line1, line2)
#                     # 求出 center 与 farthest_nodes 和 (avg_x, avg_y) 三个点的距离平均值
#                     radius = (np.linalg.norm(np.array(center) - np.array(farthest_nodes[0])) + 
#                               np.linalg.norm(np.array(center) - np.array(farthest_nodes[1])) + 
#                               np.linalg.norm(np.array(center) - np.array((avg_x, avg_y)))) / 3
                    
#                     # 将结果按照 center_x, center_y, radius, 0, 0 的形式添加进 detected_arcs 中
#                     detected_arcs.append([center[0], center[1], radius, 0, 0])
#                     pts.append((farthest_nodes[0], farthest_nodes[1], (avg_x, avg_y), center))
#                 else:
#                     print("中垂线与轮廓的交点检测失败")
#             else:
#                 print("圆弧端点检测失败")
#                 # 绘制轮廓
#                 # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
#                 # 在轮廓附近标注“圆弧端点检测失败”
#                 # cv2.putText(image, "Arc endpoints detection failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
#         except nx.NetworkXError:
#             # 处理空图或只有一个节点的情况
#             pass
    
#     return detected_arcs, pts

def arc_det(img):
    num_labels, labeled_image = cv2.connectedComponents(img)
    if num_labels <= 1:
        return None, None

    detected_arcs = []
    detected_pts = []

    for label in range(1, num_labels):
        component = np.uint8(labeled_image == label) * 255
        G = generate_graph_from_component(component)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        try:
            all_pairs_dists = dict(nx.all_pairs_shortest_path_length(G))
            max_dist = 0
            farthest_nodes = None
            for node1, dists in all_pairs_dists.items():
                for node2, dist in dists.items():
                    if dist > max_dist:
                        max_dist = dist
                        farthest_nodes = (node1, node2)

            if farthest_nodes is not None:
                line = ((farthest_nodes[0][0], farthest_nodes[0][1]), (farthest_nodes[1][0], farthest_nodes[1][1]))
                avg_point = find_perpendicular_bisector_arc_intersection(line, contours)

                if avg_point is not None:
                    avg_x, avg_y = avg_point
                    line1 = (farthest_nodes[0], (avg_x, avg_y))
                    line2 = (farthest_nodes[1], (avg_x, avg_y))
                    center = find_perpendicular_bisector_intersection(line1, line2)
                    radius = (np.linalg.norm(np.array(center) - np.array(farthest_nodes[0])) + 
                              np.linalg.norm(np.array(center) - np.array(farthest_nodes[1])) + 
                              np.linalg.norm(np.array(center) - np.array((avg_x, avg_y)))) / 3

                    angle1, angle2 = find_arc_angles(component, center, radius, farthest_nodes)
                    
                    detected_arcs.append([center[0], center[1], radius, angle1, angle2])
                    detected_pts.append((farthest_nodes[0], farthest_nodes[1], (avg_x, avg_y), center))
                else:
                    print("中垂线与轮廓的交点检测失败")
            else:
                print("圆弧端点检测失败")
                        
        except nx.NetworkXError:
            pass
    
    return detected_arcs, detected_pts

# 构件圆弧检测
def component_arc_det_test(img_path_list):
    intermediate_res = {}
    res = {}
    res_pts = {}

    for img_path in img_path_list:
        intermediate_res[img_path] = []

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to load image {img_path}")
            continue
        
        # todo: 提升分辨率两倍是为了，腐蚀时能够区分构件和辅助信息。但是使用插值法提升分辨率，效果不是很好，对于小圆弧造成一定形变，影响检测效果。
        img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 两次腐蚀+一次膨胀
        kernel_size = 3
        eroded = cv2.erode(binary, np.ones((kernel_size, kernel_size), np.uint8), iterations=2)
        eroded = remove_tiny(eroded)

        dilated_1 = cv2.dilate(eroded, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
        eroded = dilated_1

        # 去除箭头
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

        # 去除构件外元素
        mode = cv2.RETR_EXTERNAL
        contours, _ = cv2.findContours(arrows_removed, mode=mode, method=cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 3000:
                cv2.fillPoly(arrows_removed, [contour], 0)
        outside_removed = arrows_removed

        # 去除圆形
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

        # 无关元素去除效果可视化
        intermediate_res[img_path].append(circles_removed)


        # 腐蚀
        kernel_size = 3
        eroded = cv2.erode(circles_removed, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
        circles_removed = eroded

        # 腐蚀结果可视化
        intermediate_res[img_path].append(eroded.copy())


        # 去除直线（仅使用检测水平线和垂直线的霍夫直线检测）（其他直线检测算法容易去除圆弧上线段）
        lines = cv2.HoughLinesP(circles_removed, rho=1, theta=np.pi/2, threshold=10, minLineLength=30, maxLineGap=10)
        lines = np.squeeze(lines)
        if lines is not None:
            for idx, line in enumerate(lines):
                x1, y1, x2, y2 = list(map(int, line))
                cv2.line(circles_removed, (x1, y1), (x2, y2), (0, 0, 0), 3)
        else:
            print(f"No component lines detected for image {img_path} using method hough.")
        lines_removed = circles_removed

        remove_tiny(lines_removed, area_threshold=15)
        
        # 直线去除效果可视化
        intermediate_res[img_path].append(lines_removed)


        # todo: 缺少一些保证圆弧完整性的步骤




        # todo: 增加对不含圆弧构件图像的处理逻辑。此时，arcs应该是空的
        # 圆形坐标x，y，半径r，起始角度a，终止角度b
        arcs, pts = arc_det(lines_removed)
        if arcs:
            arcs = [[x / 2 for x in sublist[:3]] + sublist[3:] for sublist in arcs]
        res[img_path] = arcs
        if pts:
            pts = [[(x[0] / 2, x[1] / 2) for x in pt] for pt in pts]
        res_pts[img_path] = pts
        
        

    return res, res_pts, intermediate_res

def component_arc_det(img_path_list):
    res = {}

    for img_path in img_path_list:

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to load image {img_path}")
            continue
        
        # todo: 提升分辨率两倍是为了，腐蚀时能够区分构件和辅助信息。但是使用插值法提升分辨率，效果不是很好，对于小圆弧造成一定形变，影响检测效果。
        img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 两次腐蚀+一次膨胀
        kernel_size = 3
        eroded = cv2.erode(binary, np.ones((kernel_size, kernel_size), np.uint8), iterations=2)
        eroded = remove_tiny(eroded)

        dilated_1 = cv2.dilate(eroded, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
        eroded = dilated_1

        # 去除箭头
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

        # 去除构件外元素
        mode = cv2.RETR_EXTERNAL
        contours, _ = cv2.findContours(arrows_removed, mode=mode, method=cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 3000:
                cv2.fillPoly(arrows_removed, [contour], 0)
        outside_removed = arrows_removed

        # 去除圆形
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


        # 腐蚀
        kernel_size = 3
        eroded = cv2.erode(circles_removed, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
        circles_removed = eroded



        # 去除直线（仅使用检测水平线和垂直线的霍夫直线检测）（其他直线检测算法容易去除圆弧上线段）
        lines = cv2.HoughLinesP(circles_removed, rho=1, theta=np.pi/2, threshold=10, minLineLength=30, maxLineGap=10)
        lines = np.squeeze(lines)
        if lines is not None:
            for idx, line in enumerate(lines):
                x1, y1, x2, y2 = list(map(int, line))
                cv2.line(circles_removed, (x1, y1), (x2, y2), (0, 0, 0), 3)
        else:
            print(f"No component lines detected for image {img_path} using method hough.")
        lines_removed = circles_removed

        remove_tiny(lines_removed, area_threshold=15)
        



        # todo: 缺少一些保证圆弧完整性的步骤




        # todo: 增加对不含圆弧构件图像的处理逻辑。此时，arcs应该是空的
        # 圆形坐标x，y，半径r，起始角度a，终止角度b
        arcs, pts = arc_det(lines_removed)
        if arcs:
            arcs = [[x / 2 for x in sublist[:3]] + sublist[3:] for sublist in arcs]
        res[img_path] = arcs
        
        

    return res

# 代码测试
# def test_component_arc_det():
#     log_root = '/home/chenzhuofan/project_que/pipeline_jingzhi/logs'
#     logname_1 = '构件圆弧检测-去除直线测试'
#     logname_2 = '构件圆弧检测-检测测试'
#     if not os.path.exists(os.path.join(log_root, logname_1)):
#         os.makedirs(os.path.join(log_root, logname_1))
#     if not os.path.exists(os.path.join(log_root, logname_2)):
#         os.makedirs(os.path.join(log_root, logname_2))


#     img_dir = '/home/chenzhuofan/project_que/pipeline_jingzhi/data/精智demo展示案例备选二'
#     img_path_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]


#     res, res_pts, intermediate_res = component_arc_det(img_path_list)


#     for img_path, temp in intermediate_res.items():
#         baseneame = os.path.basename(img_path)
#         prefix, suffix = os.path.splitext(baseneame)
#         img = cv2.imread(img_path)

        
#         cv2.imwrite(os.path.join(log_root, logname_1, prefix + '_circle_removed' + suffix), temp[0])
#         cv2.imwrite(os.path.join(log_root, logname_1, prefix + '_eroded' + suffix), temp[1])
#         cv2.imwrite(os.path.join(log_root, logname_1, prefix + '_line_removed' + suffix), temp[2])

#     for img_path, arcs in res.items():
#         baseneame = os.path.basename(img_path)
#         prefix, suffix = os.path.splitext(baseneame)
#         img_red = cv2.imread(img_path)
#         img_random = img_red.copy()

#         if arcs is None:
#             continue
#         for arc in arcs:
#             center_x, center_y, radius, start_angle, end_angle = arc
#             # 绘制红色圆弧
#             cv2.circle(img_red, (int(center_x), int(center_y)), int(radius), (0, 0, 255), 2)
#             # 绘制随机颜色圆弧
#             random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#             cv2.circle(img_random, (int(center_x), int(center_y)), int(radius), random_color, 2)
        
#         # 保存红色圆弧结果
#         red_filename = os.path.join(log_root, logname_2, prefix + '_detected_red' + suffix)
#         cv2.imwrite(red_filename, img_red)
        
#         # 保存随机颜色圆弧结果
#         random_filename = os.path.join(log_root, logname_2, prefix + '_detected_random' + suffix)
#         cv2.imwrite(random_filename, img_random)

#     for img_path, points in res_pts.items():
#         baseneame = os.path.basename(img_path)
#         prefix, suffix = os.path.splitext(baseneame)
#         img_pts = cv2.imread(img_path)
        
#         if points is None:
#             continue
        
#         for point_set in points:
#             random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # 每个点集使用相同的随机颜色
#             for point in point_set:
#                 cv2.circle(img_pts, (int(point[0]), int(point[1])), 1, random_color, -1)
        
#         # 保存 pts 结果
#         pts_filename = os.path.join(log_root, logname_2, prefix + '_detected_pts' + suffix)
#         cv2.imwrite(pts_filename, img_pts)

def test_component_arc_det():
    log_root = '/home/chenzhuofan/project_que/pipeline_jingzhi/logs'
    logname_1 = '构件圆弧检测-去除直线测试'
    logname_2 = '构件圆弧检测-检测测试'
    if not os.path.exists(os.path.join(log_root, logname_1)):
        os.makedirs(os.path.join(log_root, logname_1))
    if not os.path.exists(os.path.join(log_root, logname_2)):
        os.makedirs(os.path.join(log_root, logname_2))

    img_dir = '/home/chenzhuofan/project_que/pipeline_jingzhi/data/精智demo展示案例备选二'
    img_path_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]

    res, res_pts, intermediate_res = component_arc_det_test(img_path_list)

    for img_path, temp in intermediate_res.items():
        baseneame = os.path.basename(img_path)
        prefix, suffix = os.path.splitext(baseneame)
        img = cv2.imread(img_path)

        cv2.imwrite(os.path.join(log_root, logname_1, prefix + '_circle_removed' + suffix), temp[0])
        cv2.imwrite(os.path.join(log_root, logname_1, prefix + '_eroded' + suffix), temp[1])
        cv2.imwrite(os.path.join(log_root, logname_1, prefix + '_line_removed' + suffix), temp[2])

    for img_path, arcs in res.items():
        baseneame = os.path.basename(img_path)
        prefix, suffix = os.path.splitext(baseneame)
        img_red = cv2.imread(img_path)
        img_random = img_red.copy()
        img_red_circle = img_red.copy()
        img_random_circle = img_red.copy()

        if arcs is None:
            continue

        for arc in arcs:
            center_x, center_y, radius, start_angle, end_angle = arc
            # 绘制红色圆弧
            cv2.ellipse(img_red, (int(center_x), int(center_y)), (int(radius), int(radius)), 0, start_angle, end_angle, (0, 0, 255), 2)
            # 绘制随机颜色圆弧
            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.ellipse(img_random, (int(center_x), int(center_y)), (int(radius), int(radius)), 0, start_angle, end_angle, random_color, 2)

            # 绘制红色圆形
            cv2.circle(img_red_circle, (int(center_x), int(center_y)), int(radius), (0, 0, 255), 2)
            # 绘制随机颜色圆形
            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.circle(img_random_circle, (int(center_x), int(center_y)), int(radius), random_color, 2)

        # 保存红色圆弧结果
        red_filename = os.path.join(log_root, logname_2, prefix + '_detected_red' + suffix)
        cv2.imwrite(red_filename, img_red)

        # 保存随机颜色圆弧结果
        random_filename = os.path.join(log_root, logname_2, prefix + '_detected_random' + suffix)
        cv2.imwrite(random_filename, img_random)

        red_filename = os.path.join(log_root, logname_2, prefix + '_detected_red_circle' + suffix)
        cv2.imwrite(red_filename, img_red_circle)

        random_filename = os.path.join(log_root, logname_2, prefix + '_detected_random_circle' + suffix)
        cv2.imwrite(random_filename, img_random)

    for img_path, points in res_pts.items():
        baseneame = os.path.basename(img_path)
        prefix, suffix = os.path.splitext(baseneame)
        img_pts = cv2.imread(img_path)

        if points is None:
            continue

        for point_set in points:
            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # 每个点集使用相同的随机颜色
            for point in point_set:
                cv2.circle(img_pts, (int(point[0]), int(point[1])), 1, random_color, -1)

        # 保存 pts 结果
        pts_filename = os.path.join(log_root, logname_2, prefix + '_detected_pts' + suffix)
        cv2.imwrite(pts_filename, img_pts)



if __name__ == "__main__":
    test_component_arc_det()
