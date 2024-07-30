import sys
sys.path.append("/home/chenzhuofan/pipeline/project/fudanVIA")

import json
import math
import numpy as np
from PIL import Image

import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 初始化TF-IDF向量器
vectorizer = TfidfVectorizer()




def calculate_distance(x1, y1, x2, y2):
    """计算两点之间的欧几里得距离"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def line_value_pair(lines, keywords):
    key_pairs = []
    for line in lines:
        line_centroid = [(line[0] + line[2]) // 2, (line[1] + line[3]) // 2]
        line2text = []
        for keyword in keywords:
            text_ctr = [(keyword["bbox"][0] + keyword["bbox"][2]) // 2, (keyword["bbox"][1] + keyword["bbox"][3]) // 2]
            # 计算欧氏距离
            c_dist = math.sqrt((line_centroid[0] - text_ctr[0]) ** 2 + (line_centroid[1] - text_ctr[1]) ** 2)

            # 调整分数以优先匹配线段上方和左边的文本
            # 判断文本是否在线段的左边或上方
            is_above_or_left = (text_ctr[1] < line_centroid[1]) or (text_ctr[0] < line_centroid[0])
            # 如果在上方或左边，减少距离分数以优先匹配
            if is_above_or_left:
                c_dist *= 0.5  # 可以调整这个系数来控制优先级的程度
            length_pair = len(keyword["text"]) / line[4]
            line2text.append([c_dist, length_pair])
        key_pairs.append(line2text)

    f1, f2 = [], []
    for line_pair in key_pairs:
        idx = min(enumerate(line_pair), key=lambda x: x[1][0])[0]
        f1.append(line_pair[idx][0])
        f2.append(line_pair[idx][1])
    mean = [np.mean(f1), np.mean(f2)]
    std = [np.std(f1), np.std(f2)]

    nf1, nf2 = [], []
    for x in f1:
        z_score = (x - mean[0]) / std[0] if std[0] != 0 else 0
        if -1 <= z_score <= 1:
            nf1.append(x)
    for x in f2:
        z_score = (x - mean[1]) / std[1] if std[1] != 0 else 0
        if -1 <= z_score <= 1:
            nf2.append(x)
    mean = [np.mean(nf1), np.mean(nf2)]
    std = [np.std(nf1), np.std(nf2)]

    scores = []
    for line_pair in key_pairs:
        line_scores = []
        for pair in line_pair:
            z_score_f1 = (pair[0] - mean[0]) / std[0] if std[0] != 0 else 0
            z_score_f2 = (pair[1] - mean[1]) / std[1] if std[1] != 0 else 0
            score = 0.8 * abs(z_score_f1) + 0.2 * abs(z_score_f2)
            line_scores.append(score)
        scores.append(line_scores)

    best_pair = [None] * len(lines)
    for i in range(len(lines)):
        min_val, min_idx = min((val, idx) for idx, val in enumerate(scores[i]))
        if min_val > 50:
            continue
        best_pair[i] = min_idx
        for j in range(len(lines)):
            scores[j][min_idx] = float('inf')

    return best_pair
# def line_value_pair(lines, keywords):
#     key_pairs = []
#     for line in lines:
#         centroid = [(line[0] + line[2]) // 2, (line[1] + line[3]) // 2]
#         line2text = []
#         for keyword in keywords:
#             text_ctr = [(keyword["bbox"][0] + keyword["bbox"][2]) // 2, (keyword["bbox"][1] + keyword["bbox"][3]) // 2]
#             c_dist = math.sqrt((centroid[0] - text_ctr[0]) ** 2 + (centroid[1] - text_ctr[1]) ** 2)
#             length_pair = len(keyword["text"]) / line[4]
#             line2text.append([c_dist, length_pair])
#         key_pairs.append(line2text)

#     f1, f2 = [], []
#     for line_pair in key_pairs:
#         idx = min(enumerate(line_pair), key=lambda x: x[1][0])[0]
#         f1.append(line_pair[idx][0])
#         f2.append(line_pair[idx][1])
#     mean = [np.mean(f1), np.mean(f2)]
#     std = [np.std(f1), np.std(f2)]

#     nf1, nf2 = [], []
#     for x in f1:
#         z_score = (x - mean[0]) / std[0] if std[0] != 0 else 0
#         if -1 <= z_score <= 1:
#             nf1.append(x)
#     for x in f2:
#         z_score = (x - mean[1]) / std[1] if std[1] != 0 else 0
#         if -1 <= z_score <= 1:
#             nf2.append(x)
#     mean = [np.mean(nf1), np.mean(nf2)]
#     std = [np.std(nf1), np.std(nf2)]

#     scores = []
#     for line_pair in key_pairs:
#         line_scores = []
#         for pair in line_pair:
#             z_score_f1 = (pair[0] - mean[0]) / std[0] if std[0] != 0 else 0
#             z_score_f2 = (pair[1] - mean[1]) / std[1] if std[1] != 0 else 0
#             score = 0.8 * abs(z_score_f1) + 0.2 * abs(z_score_f2)
#             line_scores.append(score)
#         scores.append(line_scores)

#     best_pair = [None] * len(lines)
#     for i in range(len(lines)):
#         min_val, min_idx = min((val, idx) for idx, val in enumerate(scores[i]))
#         if min_val > 50:
#             continue
#         best_pair[i] = min_idx
#         for j in range(len(lines)):
#             scores[j][min_idx] = float('inf')

#     return best_pair


def min_mat_value(v):
    min_val = float('inf')
    min_idx = (0, 0)

    for i in range(len(v)):
        for j in range(len(v[i])):
            if v[i][j] < min_val:
                min_val = v[i][j]
                min_idx = (i, j)

    return min_val, min_idx

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size

def line_value_pair_old(lines, keywords):
    key_pairs = []
    for line in lines:
        centeroid = [(line[0] + line[2]) // 2, (line[1] + line[3]) // 2]
        line2text = []
        for keyword in keywords:
            text_ctr = [(keyword["bbox"][0] + keyword["bbox"][2]) // 2, (keyword["bbox"][1] + keyword["bbox"][3]) // 2]
            c_dist = math.sqrt((centeroid[0] - text_ctr[0]) ** 2 + (centeroid[1] - text_ctr[1]) ** 2)
            length_pair = float(keyword["text"]) / line[4]
            line2text.append([c_dist, length_pair])
        key_pairs.append(line2text)

    f1, f2 = [], []
    for line_pair in key_pairs:
        idx = min(enumerate(line_pair), key=lambda x: x[1][0])[0]
        f1.append(line_pair[idx][0])
        f2.append(line_pair[idx][1])
    mean = [np.mean(f1), np.mean(f2)]
    std = [np.std(f1), np.std(f2)]
    print("f1:", f1)
    print("f2:", f2)
    print("std1:", std)
    nf1, nf2 = [], []
    for x in f1:
        if x == mean[0]:
            z_score = 0
        else:
            z_score = (x - mean[0]) / std[0]
        if -1 <= z_score <= 1:
            nf1.append(x)
    for x in f2:
        if x == mean[1]:
            z_score = 0
        else:
            z_score = (x - mean[1]) / std[1]
        if -1 <= z_score <= 1:
            nf2.append(x)
    mean = [np.mean(nf1), np.mean(nf2)]
    std = [np.std(nf1), np.std(nf2)]
    print("std2:", std)

    scores = []
    for line_pair in key_pairs:
        line_scores = []
        for pair in line_pair:
            if pair[0] == mean[0]:
                z_score_f1 = 0
            else:
                z_score_f1 = (pair[0] - mean[0]) / std[0]
            if pair[1] == mean[1]:
                z_score_f2 = 0
            else:
                z_score_f2 = (pair[1] - mean[1]) / std[1]
            score = 0.8 * abs(z_score_f1) + 0.2 * abs(z_score_f2)
            line_scores.append(score)
        scores.append(line_scores)

    best_pair = [None] * len(lines)
    for i in range(len(lines)):
        min_val, min_idx = min_mat_value(scores)
        if min_val > 50:
            break
        best_pair[min_idx[0]] = min_idx[1]
        for j in range(len(keywords)):
            scores[min_idx[0]][i] = float('inf')
        for j in range(len(lines)):
            scores[j][min_idx[1]] = float('inf')

    return best_pair

# import math
# import numpy as np

# def line_value_pair(lines, keywords):
#     key_pairs = []
#     filtered_lines = [line for line in lines if line[4] != 0]  # Exclude lines with zero length
#     for line in filtered_lines:
#         centeroid = [(line[0] + line[2]) // 2, (line[1] + line[3]) // 2]
#         line2text = []
#         for keyword in keywords:
#             text_ctr = [(keyword["bbox"][0] + keyword["bbox"][2]) // 2, (keyword["bbox"][1] + keyword["bbox"][3]) // 2]
#             c_dist = math.sqrt((centeroid[0] - text_ctr[0]) ** 2 + (centeroid[1] - text_ctr[1]) ** 2)
#             if line[4] != 0:  # Additional safeguard
#                 length_pair = float(keyword["text"]) / line[4]
#                 line2text.append([c_dist, length_pair])
#         key_pairs.append(line2text)

#     f1, f2 = [], []
#     for line_pair in key_pairs:
#         idx = min(enumerate(line_pair), key=lambda x: x[1][0])[0]
#         f1.append(line_pair[idx][0])
#         f2.append(line_pair[idx][1])
#     mean = [np.mean(f1), np.mean(f2)]
#     std = [np.std(f1), np.std(f2)]
#     print("f1:", f1)
#     print("f2:", f2)
#     print("std1:", std)
#     nf1, nf2 = [], []
#     for x in f1:
#         if x == mean[0]:
#             z_score = 0
#         else:
#             z_score = (x - mean[0]) / std[0]
#         if -1 <= z_score <= 1:
#             nf1.append(x)
#     for x in f2:
#         if x == mean[1]:
#             z_score = 0
#         else:
#             z_score = (x - mean[1]) / std[1]
#         if -1 <= z_score <= 1:
#             nf2.append(x)
#     mean = [np.mean(nf1), np.mean(nf2)]
#     std = [np.std(nf1), np.std(nf2)]
#     print("std2:", std)

#     scores = []
#     for line_pair in key_pairs:
#         line_scores = []
#         for pair in line_pair:
#             if pair[0] == mean[0]:
#                 z_score_f1 = 0
#             else:
#                 z_score_f1 = (pair[0] - mean[0]) / std[0]
#             if pair[1] == mean[1]:
#                 z_score_f2 = 0
#             else:
#                 z_score_f2 = (pair[1] - mean[1]) / std[1]
#             score = 0.8 * abs(z_score_f1) + 0.2 * abs(z_score_f2)
#             line_scores.append(score)
#         scores.append(line_scores)

#     best_pair = [None] * len(filtered_lines)
#     for i in range(len(filtered_lines)):
#         min_val, min_idx = min((val, idx) for idx, val in enumerate(scores[i]))
#         if min_val > 50:
#             continue
#         best_pair[i] = min_idx
#         for j in range(len(keywords)):
#             scores[i][j] = float('inf')
#         for j in range(len(filtered_lines)):
#             scores[j][min_idx] = float('inf')

#     return best_pair



# def pair_preprocess(splited_lines, filtered_texts, best_pairs):
#     scale_features = []
#     text_features = []
#     for i, splited_line in enumerate(splited_lines):
#         x1, y1, x2, y2, _ = splited_line
#         if best_pairs[i] is not None:
#             scale_features.append(
#                 {'value': filtered_texts[best_pairs[i]]["text"], 'pts': [x1, y1, x2, y2], 'length': _})
#             text_features.append(filtered_texts[best_pairs[i]])

#     return scale_features, text_features

def pair_preprocess(splited_lines, filtered_texts, best_pairs):
    scale_features = []
    text_features = []
    for i, splited_line in enumerate(splited_lines):
        x1, y1, x2, y2, _ = splited_line
        if best_pairs[i] is not None and best_pairs[i] < len(filtered_texts):
            # 确保 best_pairs[i] 是一个有效的索引
            scale_features.append(
                {'value': filtered_texts[best_pairs[i]]["text"], 'pts': [x1, y1, x2, y2], 'length': _})
            text_features.append(filtered_texts[best_pairs[i]])
    return scale_features, text_features


# class LineValuePairModule(Module):
#     def __init__(self, input_line, output_line, control_line=None, run_num_threads=2, name=None):
#         super().__init__(input_line, output_line, control_line=control_line, run_num_thread=run_num_threads, name=name)
#         self.name = name

#     def on_terminate(self):
#         print(f'{self.name} terminated')

#     def run(self, input_package):
#         splited_line_list = input_package['splited_line_list']
#         filtered_text_list = input_package['filtered_text_list']
#         filtered_lines = [line for line in splited_line_list if line[4] >= 10]

#         b_pairs = line_value_pair(filtered_lines, filtered_text_list)
#         print(f'b_pairs:{b_pairs}')
#         scale_feature_list, text_feature_list = pair_preprocess(filtered_lines, filtered_text_list, b_pairs)

#         output_package = input_package.copy()
#         output_package["scale_feature_list"] = scale_feature_list
#         output_package["text_feature_list"] = text_feature_list

        
#         # # 将 output_package 转换为 JSON 字符串
#         # output_json = json.dumps(output_package, ensure_ascii=False, indent=4)

#         # # 将 JSON 字符串写入文件
#         # with open('/home/chenzhuofan/pipeline/project/vis/output_package.txt', 'w', encoding='utf-8') as f:
#         #     f.write(output_json)

#         return output_package
def run(input_package):
    splited_line_list = input_package['splited_line_list']
    filtered_text_list = input_package['filtered_text_list']
    filtered_lines = [line for line in splited_line_list if line[4] >= 10]
    
    b_pairs = line_value_pair(filtered_lines, filtered_text_list)
    print(f'b_pairs:{b_pairs}')
    scale_feature_list, text_feature_list = pair_preprocess(filtered_lines, filtered_text_list, b_pairs)
    
    output_package = input_package.copy()
    output_package["scale_feature_list"] = scale_feature_list
    output_package["text_feature_list"] = text_feature_list
    
    
    # # 将 output_package 转换为 JSON 字符串
    # output_json = json.dumps(output_package, ensure_ascii=False, indent=4)
    
    # # 将 JSON 字符串写入文件
    # with open('/home/chenzhuofan/pipeline/project/vis/output_package.txt', 'w', encoding='utf-8') as f:
    #     f.write(output_json)
    
    return output_package