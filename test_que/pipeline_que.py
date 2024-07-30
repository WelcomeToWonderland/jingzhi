### pipeline que 

import sys
sys.path.append("./modules")
import arrow_det_modules
from modules.circle_det_module import component_circle_det
from modules.line_det_module import annotation_line_det, component_line_det
from modules.arc_det_module import component_arc_det
import line_split_module
# from modules.text_det_rec_modules import Text_det
from modules.line_value_pair import process_scale_features
import cv2
import os
import random
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pprint
import json
import requests
import base64
import re

def image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

def rotate_image_90(image):
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def get_baidu_access_token(api_key, secret_key):
    token_url = 'https://aip.baidubce.com/oauth/2.0/token'
    params = {
        'grant_type': 'client_credentials',
        'client_id': api_key,
        'client_secret': secret_key
    }

    response = requests.post(token_url, params=params)
    access_token = response.json().get('access_token')
    if not access_token:
        raise ValueError("Failed to obtain access token")
    return access_token

def perform_ocr(image_base64, access_token):
    baidu_ocr_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate"
    params = {"access_token": access_token}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"image": image_base64}

    response = requests.post(baidu_ocr_url, params=params, headers=headers, data=data)
    if response.status_code != 200:
        raise ValueError(f"Error: {response.status_code}, {response.text}")
    
    return response.json()

def adjust_rotated_coordinates(rotated_result, original_width, original_height):
    adjusted_results = []
    for item in rotated_result:
        adjusted_location = {
            'top': original_height - item['location']['left'] - item['location']['width'],
            'left': item['location']['top'],
            'width': item['location']['height'],
            'height': item['location']['width']
        }

        adjusted_item = {
            'words': item['words'],
            'location': adjusted_location
        }
        adjusted_results.append(adjusted_item)
    return adjusted_results

def filter_results(ocr_results, apply_width_height_filter=True):
    filtered_results = []
    for item in ocr_results['words_result']:
        text = item['words']
        location = item['location']
        
        if not (text.startswith('0') or text.startswith('O') or re.match(r"^00", text)) and re.search(r'\d', text):
            if apply_width_height_filter:
                is_two_digit = re.match(r'\d{2}', text)
                
                if text.startswith("334") and len(text) > 3:
                    split_texts = ["334", "216", "27"]
                    for i, split_text in enumerate(split_texts):
                        split_width = location['width'] // len(split_texts)
                        split_location = {
                            'top': location['top'],
                            'left': location['left'] + split_width * i,
                            'width': split_width,
                            'height': location['height']
                        }
                        filtered_results.append({'words': split_text, 'location': split_location})
                elif item['location']['width'] > item['location']['height'] or is_two_digit:
                    filtered_results.append(item)
            else:
                filtered_results.append(item)
    return filtered_results

def format_results(results):
    formatted_results = []
    for item in results:
        text = item['words']
        bbox = [
            item['location']['left'],
            item['location']['top'],
            item['location']['left'] + item['location']['width'],
            item['location']['top'] + item['location']['height']
        ]
        formatted_results.append({'text': text, 'bbox': bbox})
    return formatted_results


def process_images(image_path_list, api_key, secret_key):
    access_token = get_baidu_access_token(api_key, secret_key)
    ocr_results = {}

    for img_path in image_path_list:
        original_image = cv2.imread(img_path)
        original_image_base64 = image_to_base64(original_image)
        original_height, original_width = original_image.shape[:2]

        original_ocr_result = perform_ocr(original_image_base64, access_token)
        if 'words_result' not in original_ocr_result:
            print(f"OCR failed or found no text for image: {img_path}")
            ocr_results[img_path] = []
            continue

        original_filtered_results = filter_results(original_ocr_result, apply_width_height_filter=False)

        rotated_image = rotate_image_90(original_image)
        rotated_image_base64 = image_to_base64(rotated_image)
        rotated_ocr_result = perform_ocr(rotated_image_base64, access_token)
        if 'words_result' not in rotated_ocr_result:
            print(f"OCR failed or found no text for image: {img_path}")
            ocr_results[img_path] = original_filtered_results
            continue

        rotated_filtered_results = filter_results(rotated_ocr_result)
        adjusted_rotated_results = adjust_rotated_coordinates(rotated_filtered_results, original_width, original_height)

        combined_ocr_result = original_filtered_results + adjusted_rotated_results
        formatted_combined_ocr_result = format_results(combined_ocr_result)

        ocr_results[img_path] = formatted_combined_ocr_result

    return ocr_results

api_key = 'EgrYRRm2UWT0Wv3ZDKStQDD2'
secret_key = 'vSm1Y6qKjwbgAqAmG0DISzBmbtjmvodU'


def visualize_on_image(image_path, lines, output_path, font_path):
    # 加载图片
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 24)  # 可以调整字体大小以适应视觉效果
    
    # 绘制线段并旁边显示文本
    for line in lines:
        x1, y1, x2, y2 = line['pts']
        draw.line((x1, y1, x2, y2), fill='red', width=3)

        # 计算文本位置：简单地取两个端点的中点
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        text_position = (mid_x, mid_y)
        
        # 绘制文本，文本是`value`
        draw.text(text_position, line['value'], fill='black', font=font)
    
    # 保存图片
    img.save(output_path)
    print(f"Image saved to {output_path}")


def add_length_to_lines(lines):
    result = []
    for line in lines:
        x1, y1, x2, y2 = line
        # 计算线段的长度
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # 将长度添加到线段的信息中，并格式化输出
        result.append([x1, y1, x2, y2, round(length, 1)])
    return result


def calculate_ratio(consolidated_dict):
    """
    计算所有图像的 scale_feature_list 中每条线段的 feature_value 与线段长度的比值的总体平均值。
    
    :param consolidated_dict: 整合后的字典，包含每个图像的 scale_feature_list
    :return: 所有图像的 feature_value 与线段长度比值的总体平均值
    """
    total_ratio = 0
    count = 0
    
    for img_path, data in consolidated_dict.items():
        scale_feature_list = data.get('scale_feature_list', [])
        for line in scale_feature_list:
            x1, y1, x2, y2 = line['pts']
            feature_value = line['value']

            # 计算线段长度
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # 计算 feature_value 与 length 的比值
            try:
                ratio = float(feature_value) / length
            except ValueError:
                print(f"Error: Invalid value for feature_value {feature_value} for image {img_path}")
                continue
            except ZeroDivisionError:
                print(f"Error: Zero length for line in image {img_path}")
                continue

            total_ratio += ratio
            count += 1
    
    # 计算总体平均比值
    if count > 0:
        overall_average_ratio = total_ratio / count
    else:
        overall_average_ratio = None  # 如果没有有效的数据，返回 None 或其他适当的值
    
    return overall_average_ratio



def scale_components(consolidated_dict, scale_ratio):
    """
    使用给定的缩放比率对 component_lines、component_circles 和 component_arcs 进行缩放。
    
    :param consolidated_dict: 整合后的字典，包含每个图像的 component_lines、component_circles 和 component_arcs
    :param scale_ratio: 计算得到的 feature_value 与线段长度比值的总体平均值
    :return: 包含缩放后的 redrawed_component_lines、redrawed_component_circles 和 redrawed_component_arcs 的字典
    """
    for img_path, data in consolidated_dict.items():
        # 缩放 component_lines
        component_lines = data.get('component_lines', [])
        redrawed_component_lines = [
            [x1 * scale_ratio, y1 * scale_ratio, x2 * scale_ratio, y2 * scale_ratio]
            for x1, y1, x2, y2 in component_lines
        ]
        data['redrawed_component_lines'] = redrawed_component_lines

        # 缩放 component_circles
        component_circles = data.get('component_circles', [])
        redrawed_component_circles = [
            [cx * scale_ratio, cy * scale_ratio, r * scale_ratio]
            for cx, cy, r in component_circles
        ]
        data['redrawed_component_circles'] = redrawed_component_circles

        # 缩放 component_arcs
        component_arcs = data.get('component_arcs', [])
        if component_arcs:
            redrawed_component_arcs = [
                [cx * scale_ratio, cy * scale_ratio, r * scale_ratio, start_angle, end_angle]
                for cx, cy, r, start_angle, end_angle in component_arcs
            ]
            data['redrawed_component_arcs'] = redrawed_component_arcs
        else:
            data['redrawed_component_arcs'] = []

    return consolidated_dict




def visualize_original_components(consolidated_dict, log_root):
    """
    使用OpenCV在图像上绘制原始的 component_lines、component_circles 和 component_arcs，并将结果保存到指定的日志文件夹中。
    
    :param consolidated_dict: 整合后的字典，包含每个图像的 component_lines、component_circles 和 component_arcs
    :param log_root: 日志文件夹路径
    """
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    
    for img_path, data in consolidated_dict.items():
        # 加载图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to load image {img_path}")
            continue
        
        # 绘制原始的 component_lines
        component_lines = data.get('component_lines', [])
        for line in component_lines:
            x1, y1, x2, y2 = map(int, line)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色线条，宽度为3

        # 绘制原始的 component_circles
        component_circles = data.get('component_circles', [])
        for circle in component_circles:
            cx, cy, r = map(int, circle)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), 2)  # 红色圆圈，宽度为3

        # 绘制原始的 component_arcs
        component_arcs = data.get('component_arcs', [])
        if component_arcs:
            for arc in component_arcs:
                cx, cy, r, start_angle, end_angle = arc  # 直接解包
                cx, cy, r = map(int, (cx, cy, r))  # 将前三个元素转换为整数
                start_angle, end_angle = float(start_angle), float(end_angle)  # 确保角度为浮点数
                cv2.ellipse(img, (cx, cy), (r, r), 0, start_angle, end_angle, (0, 0, 255), 2)  # 红色弧线，宽度为3
        
        # 生成可视化输出路径
        base_name, ext = os.path.splitext(os.path.basename(img_path))
        output_path = os.path.join(log_root, f"{base_name}_original_components{ext}")
        
        # 保存图片
        cv2.imwrite(output_path, img)
        print(f"Image saved to {output_path}")





def visualize_scaled_components(consolidated_dict, log_root):
    """
    使用OpenCV在图像上绘制缩放后的 component_lines、component_circles 和 component_arcs，并将结果保存到指定的日志文件夹中。
    
    :param consolidated_dict: 整合后的字典，包含每个图像的 redrawed_component_lines、redrawed_component_circles 和 redrawed_component_arcs
    :param log_root: 日志文件夹路径
    """
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    
    for img_path, data in consolidated_dict.items():
        # 加载图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to load image {img_path}")
            continue
        
        # 绘制缩放后的 redrawed_component_lines
        redrawed_component_lines = data.get('redrawed_component_lines', [])
        for line in redrawed_component_lines:
            # x1, y1, x2, y2 = line
            x1, y1, x2, y2 = map(int, line)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)  # 红色线条，宽度为3

        # 绘制缩放后的 redrawed_component_circles
        redrawed_component_circles = data.get('redrawed_component_circles', [])
        for circle in redrawed_component_circles:
            # cx, cy, r = circle
            cx, cy, r = map(int, circle)            
            cv2.circle(img, (cx, cy), r, (0, 0, 255), 1)  # 红色圆圈，宽度为3

        # 绘制缩放后的 redrawed_component_arcs
        redrawed_component_arcs = data.get('redrawed_component_arcs', [])
        if redrawed_component_arcs:
            for arc in redrawed_component_arcs:
                cx, cy, r, start_angle, end_angle = arc  # 直接解包
                cx, cy, r = map(int, (cx, cy, r))  # 将前三个元素转换为整数
                start_angle, end_angle = float(start_angle), float(end_angle)  # 确保角度为浮点数
                cv2.ellipse(img, (cx, cy), (r, r), 0, start_angle, end_angle, (0, 0, 255), 1)  # 红色弧线，宽度为3
        
        # 生成可视化输出路径
        base_name, ext = os.path.splitext(os.path.basename(img_path))
        output_path = os.path.join(log_root, f"{base_name}_scaled_components{ext}")
        
        # 保存图片
        cv2.imwrite(output_path, img)
        print(f"Image saved to {output_path}")




def convert_to_cartesian(scaled_consolidated_dict):
    """
    将 scaled_consolidated_dict 中的 redrawed_component_lines、redrawed_component_circles 和 redrawed_component_arcs 
    从像素矩阵坐标系转换为一般二维坐标系。
    
    :param scaled_consolidated_dict: 包含每个图像的 redrawed_component_lines、redrawed_component_circles 和 redrawed_component_arcs 的字典
    :return: 转换后的字典
    """
    def convert_line(line, img_height):
        x1, y1, x2, y2 = line
        return [x1, img_height - 1 - y1, x2, img_height - 1 - y2]

    def convert_circle(circle, img_height):
        cx, cy, r = circle
        return [cx, img_height - 1 - cy, r]

    def convert_arc(arc, img_height):
        cx, cy, r, start_angle, end_angle = arc
        return [cx, img_height - 1 - cy, r, start_angle, end_angle]

    converted_dict = {}

    for img_path, data in scaled_consolidated_dict.items():
        # 读取图像以获取高度
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to load image {img_path}")
            continue
        img_height = img.shape[0]

        # 转换 redrawed_component_lines
        redrawed_component_lines = data.get('redrawed_component_lines', [])
        converted_lines = [convert_line(line, img_height) for line in redrawed_component_lines]

        # 转换 redrawed_component_circles
        redrawed_component_circles = data.get('redrawed_component_circles', [])
        converted_circles = [convert_circle(circle, img_height) for circle in redrawed_component_circles]

        # 转换 redrawed_component_arcs
        redrawed_component_arcs = data.get('redrawed_component_arcs', [])
        converted_arcs = [convert_arc(arc, img_height) for arc in redrawed_component_arcs] if redrawed_component_arcs else []

        # 更新转换后的数据
        converted_dict[img_path] = {
            'redrawed_component_lines': converted_lines,
            'redrawed_component_circles': converted_circles,
            'redrawed_component_arcs': converted_arcs
        }

    return converted_dict


def visualize_converted_scaled_components(converted_dict, log_root):
    """
    使用OpenCV在与原图像相同大小的空白图像上绘制转换后的缩放组件，并将结果保存到指定的日志文件夹中。
    
    :param converted_dict: 包含每个图像的 redrawed_component_lines、redrawed_component_circles 和 redrawed_component_arcs 的字典
    :param log_root: 日志文件夹路径
    """
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    
    for img_path, data in converted_dict.items():
        # 加载原图像以获取尺寸
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"Error: Unable to load image {img_path}")
            continue
        
        # 创建与原图像相同大小的空白图像
        height, width, _ = original_img.shape
        blank_img = np.ones((height, width, 3), np.uint8) * 255  # 创建白色背景图像

        # 绘制转换后的 component_lines
        component_lines = data.get('redrawed_component_lines', [])
        for line in component_lines:
            x1, y1, x2, y2 = map(int, line)
            cv2.line(blank_img, (x1, y1), (x2, y2), (0, 0, 255), 1)  # 红色线条，宽度为1

        # 绘制转换后的 component_circles
        component_circles = data.get('redrawed_component_circles', [])
        for circle in component_circles:
            cx, cy, r = map(int, circle)
            cv2.circle(blank_img, (cx, cy), r, (0, 0, 255), 1)  # 红色圆圈，宽度为1

        # 绘制转换后的 component_arcs
        component_arcs = data.get('redrawed_component_arcs', [])
        for arc in component_arcs:
            cx, cy, r, start_angle, end_angle = arc
            cx, cy, r = int(cx), int(cy), int(r)
            start_angle, end_angle = float(start_angle), float(end_angle)
            cv2.ellipse(blank_img, (cx, cy), (r, r), 0, start_angle, end_angle, (0, 0, 255), 1)  # 红色弧线，宽度为1
        
        # 生成输出路径
        base_name, ext = os.path.splitext(os.path.basename(img_path))
        output_path = os.path.join(log_root, f"{base_name}_converted{ext}")
        
        # 保存图片
        cv2.imwrite(output_path, blank_img)
        print(f"Image saved to {output_path}")


def save_scaled_components_to_json(scaled_consolidated_dict, log_root):
    """
    将缩放后的组件保存为 JSON 文件，每个 img_path 对应一个 JSON 文件。
    
    :param scaled_consolidated_dict: 包含每个图像的缩放后组件的字典
    :param log_root: JSON 文件保存的路径
    """
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    
    for img_path, data in scaled_consolidated_dict.items():
        json_dict = {"entityList": []}
        
        # 处理缩放后的 line
        redrawed_component_lines = data.get('redrawed_component_lines', [])
        if redrawed_component_lines is not None:
            for line in redrawed_component_lines:
                x0, y0, x1, y1 = line
                json_dict["entityList"].append({
                    "type": "ZcDbLine",
                    "startPoint": [x0, y0, 0.0],
                    "endPoint": [x1, y1, 0.0],
                    "color": [255, 255, 255],
                    "layer": "基本实体",
                    "lineType": "ByLayer",
                    "lineTypeScale": 1.0,
                    "lineweight": -1,
                    "thickness": 0.0,
                    "transparency": 0
                })
        
        # 处理缩放后的 circle
        redrawed_component_circles = data.get('redrawed_component_circles', [])
        if redrawed_component_circles is not None:
            for cx, cy, r in redrawed_component_circles:
                json_dict["entityList"].append({
                    "type": "ZcDbCircle",
                    "center": [cx, cy, 0.0],
                    "radius": r,
                    "color": 256,
                    "layer": "基本实体",
                    "lineType": "ByLayer",
                    "lineTypeScale": 1.0,
                    "lineweight": -1,
                    "normal": [0.0, 0.0, 1.0],
                    "thickness": 0.0,
                    "transparency": 0
                })

        # 处理缩放后的 arc
        redrawed_component_arcs = data.get('redrawed_component_arcs', [])
        if redrawed_component_arcs is not None:
            for center_x, center_y, radius, start_angle, end_angle in redrawed_component_arcs:
                json_dict["entityList"].append({
                    "type": "ZcDbArc",
                    "center": [center_x, center_y, 0.0],
                    "radius": radius,
                    "startAngle": start_angle,
                    "endAngle": end_angle,
                    "color": 256,
                    "layer": "基本实体",
                    "lineType": "ByLayer",
                    "lineTypeScale": 1.0,
                    "lineweight": -1,
                    "normal": [0.0, 0.0, 1.0],
                    "thickness": 0.0,
                    "transparency": 0
                })
        
        # 生成 JSON 文件名和路径
        base_name, _ = os.path.splitext(os.path.basename(img_path))
        output_path = os.path.join(log_root, f"{base_name}_scaled_components.json")
        
        # 保存 JSON 文件
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_dict, json_file, ensure_ascii=False, indent=4)
        
        print(f"JSON file saved to {output_path}")


def convert_redrawed_components_to_json(converted_dict):
    """
    将输入字典中的 redrawed 元素转换成 JSON 格式，并返回一个包含所有 JSON 的列表。

    :param converted_dict: 包含重绘组件信息的字典
    :return: 包含所有 JSON 数据的列表
    """
    json_list = []

    for img_path, data in converted_dict.items():
        json_dict = {"entityList": []}
        
        # 处理重绘的线段
        redrawed_component_lines = data.get('redrawed_component_lines', [])
        for line in redrawed_component_lines:
            x0, y0, x1, y1 = line
            json_dict["entityList"].append({
                "type": "ZcDbLine",
                "startPoint": [x0, y0, 0.0],
                "endPoint": [x1, y1, 0.0],
                "color": [255, 255, 255],
                "layer": "基本实体",
                "lineType": "ByLayer",
                "lineTypeScale": 1.0,
                "lineweight": -1,
                "thickness": 0.0,
                "transparency": 0
            })

        # 处理重绘的圆
        redrawed_component_circles = data.get('redrawed_component_circles', [])
        for circle in redrawed_component_circles:
            cx, cy, r = circle
            json_dict["entityList"].append({
                "type": "ZcDbCircle",
                "center": [cx, cy, 0.0],
                "radius": r,
                "color": 256,
                "layer": "基本实体",
                "lineType": "ByLayer",
                "lineTypeScale": 1.0,
                "lineweight": -1,
                "normal": [0.0, 0.0, 1.0],
                "thickness": 0.0,
                "transparency": 0
            })

        # 处理重绘的弧线
        redrawed_component_arcs = data.get('redrawed_component_arcs', [])
        for arc in redrawed_component_arcs:
            cx, cy, r, start_angle, end_angle = arc
            json_dict["entityList"].append({
                "type": "ZcDbArc",
                "center": [cx, cy, 0.0],
                "radius": r,
                "startAngle": start_angle,
                "endAngle": end_angle,
                "color": 256,
                "layer": "基本实体",
                "lineType": "ByLayer",
                "lineTypeScale": 1.0,
                "lineweight": -1,
                "normal": [0.0, 0.0, 1.0],
                "thickness": 0.0,
                "transparency": 0
            })

        json_list.append(json_dict)

    return json_list


def pipeline(img_dir):
    img_path_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]


    # 矢量化信息提取
    component_line_list = component_line_det(img_path_list)
    component_circle_list = component_circle_det(img_path_list)
    component_arc_list = component_arc_det(img_path_list)
    

    # 获取包含尺寸线段的线段列表
    annotation_line_list = annotation_line_det(img_path_list)   
    # 箭头检测
    arrow_dic = arrow_det_modules.pipeline_arrow_det(img_path_list)
    # 直线分割
    splited_lines = line_split_module.pipeline_split_line(annotation_line_list, arrow_dic)
    processed_split_lines = {}
    for img_path in img_path_list:
        processed_split_lines[img_path] = add_length_to_lines(splited_lines.get(img_path, []))
    
    
    # 字符检测 
    ocr_result = Text_det(img_path_list)
    # 新的
    rec_results_list = process_images(img_path_list, api_key, secret_key)
    
    
    # 整合结果
    # img_path_list = component_line_list.keys()
    consolidated_dict = {}
    for img_path in img_path_list:
        consolidated_dict[img_path] = {
            "component_lines": component_line_list.get(img_path, []),
            "component_circles": component_circle_list.get(img_path, []),
            "component_arcs": component_arc_list.get(img_path, []),
            "annotation_lines": annotation_line_list.get(img_path, []),
            "arrows": arrow_dic.get(img_path, []),
            # "split_lines": splited_lines.get(img_path, []),
            "split_lines": processed_split_lines.get(img_path, []),
            # "ocr_result": ocr_result.get(img_path, [])
            "ocr_result": rec_results_list.get(img_path, [])
        }
      
    
    # 尺寸线段与尺寸数据匹配
    consolidated_dict = process_scale_features(consolidated_dict)
    
    # 计算比例
    overall_average_ratio = calculate_ratio(consolidated_dict)
    
    # 缩放构件
    scaled_consolidated_dict = scale_components(consolidated_dict, overall_average_ratio)  
    
    # 转换坐标系
    converted_dict = convert_to_cartesian(scaled_consolidated_dict)
    
    # # 结果转存为json
    # save_scaled_components_to_json(scaled_consolidated_dict, os.path.join(log_root, logname))

    return convert_redrawed_components_to_json(converted_dict)


if __name__ == "__main__":
    # img_dir = '/home/chenzhuofan/project_que/pipeline_jingzhi/data/test'
    # img_dir = '/home/chenzhuofan/project_que/pipeline_jingzhi/data/精智demo展示案例备选一'
    img_dir = './data/精智demo展示案例备选二'
    # img_dir = '/home/chenzhuofan/project_que/pipeline_jingzhi/data/HZ-HJH23050501-A-08-01-006'
    # img_dir = '/home/chenzhuofan/project_que/pipeline_jingzhi/data/HZ-HJH23050501-A-08-01-007'
    
    
    # # 定义命令行参数
    # parser = argparse.ArgumentParser(description='Pipeline for image processing')
    # parser.add_argument('--img_dir', type=str, required=True, help='Path to the image directory')
    # args = parser.parse_args()
    # # python pipeline.py --img_dir /path/to/your/image/directory
    
    # # 从命令行参数获取 img_dir
    # img_dir = args.img_dir
    
    
    # pipeline开始
    log_root = './logs'
    logname = 'pipeline_demo展示-que-坐标系转换-圆弧角度-师清-备选二-精智服务器测试'
    font_path = './Deng.ttf'
    
    
    
    
    
    img_path_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
    if not os.path.exists(os.path.join(log_root, logname)):
        os.makedirs(os.path.join(log_root, logname))
    
    
    # 矢量化信息提取
    component_line_list = component_line_det(img_path_list)
    component_circle_list = component_circle_det(img_path_list)
    component_arc_list = component_arc_det(img_path_list)
    
    pprint.pprint(component_line_list)
    pprint.pprint(component_circle_list)
    pprint.pprint(component_circle_list)
    
    
    # 获取包含尺寸线段的线段列表
    annotation_line_list = annotation_line_det(img_path_list)
    
    pprint.pprint(annotation_line_list)
    
    # 箭头检测
    arrow_dic = arrow_det_modules.pipeline_arrow_det(img_path_list)
    # 直线分割
    splited_lines = line_split_module.pipeline_split_line(annotation_line_list, arrow_dic)
    processed_split_lines = {}
    for img_path in img_path_list:
        processed_split_lines[img_path] = add_length_to_lines(splited_lines.get(img_path, []))
    
    pprint.pprint(arrow_dic)
    pprint.pprint(processed_split_lines)
    
    
    # # 字符检测 
    # ocr_result = Text_det(img_path_list)
    # 新的
    rec_results_list = process_images(img_path_list, api_key, secret_key)
    
    
    
    
    # 整合结果
    # img_path_list = component_line_list.keys()
    consolidated_dict = {}
    for img_path in img_path_list:
        # annotation_lines = annotation_line_list.get(img_path, [])
        # annotation_lines = add_length_to_lines(annotation_lines)
        consolidated_dict[img_path] = {
            "component_lines": component_line_list.get(img_path, []),
            "component_circles": component_circle_list.get(img_path, []),
            "component_arcs": component_arc_list.get(img_path, []),
            "annotation_lines": annotation_line_list.get(img_path, []),
            "arrows": arrow_dic.get(img_path, []),
            # "split_lines": splited_lines.get(img_path, []),
            "split_lines": processed_split_lines.get(img_path, []),
            # "ocr_result": ocr_result.get(img_path, [])
            "ocr_result": rec_results_list.get(img_path, [])
        }
    
    
    # 可视化原始构件
    visualize_original_components(consolidated_dict, os.path.join(log_root, logname))   
    
    # # 尺寸线段与尺寸数据匹配
    consolidated_dict = process_scale_features(consolidated_dict)
    
    # # 计算比例
    overall_average_ratio = calculate_ratio(consolidated_dict)
    
    # # 缩放构件
    scaled_consolidated_dict = scale_components(consolidated_dict, overall_average_ratio)
    visualize_scaled_components(scaled_consolidated_dict, os.path.join(log_root, logname))   
    
    # # 转换坐标系
    # converted_dict = convert_to_cartesian(scaled_consolidated_dict)
    # visualize_converted_scaled_components(converted_dict, os.path.join(log_root, logname))
    
    # # 结果转存为json
    # save_scaled_components_to_json(scaled_consolidated_dict, os.path.join(log_root, logname))
    
