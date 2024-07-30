from paddleocr import PaddleOCR,draw_ocr
from PIL import Image
import numpy as np
import re
import cv2


def max_num(n0,n1,n2,n3):
    temp = max(n0[0][0][1],n1[0][0][1],n2[0][0][1],n3[0][0][1])
    if temp==n0[0][0][1]:
        return n0[0][0][0]
    elif temp==n1[0][0][1]:
        return n1[0][0][0]
    elif temp==n2[0][0][1]:
        return n2[0][0][0]
    else :
        return n3[0][0][0]

def Text_det(img_folder_path):
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", cls=True, det_model_dir='ocr_model/det3', rec_model_dir='ocr_model/infer')
    # final_result = []
    final_result = {}
    for index in range(len(img_folder_path)):
        img_path = img_folder_path[index]
        ocr_results = ocr.ocr(img_path,cls = True)
        image = Image.open(img_path).convert('RGB')
        crop_img_list_1 = []
        crop_img_list_2 = []
        crop_img_list_3 = []
        crop_img_list_4 = []
        point_list = []
        rec_results_list = []
        rectangule_list = []
        # print(ocr_results)
        if ocr_results[0] is not None:
            for line in ocr_results:
                for box in line:
                    points = np.array(box[0]).astype(np.int32)
                    rectangule =[points[0],points[1],points[2],points[3]]
                    rectangule_list.append(rectangule)
                    x_min, x_max = min(points[:, 0]), max(points[:, 0])
                    y_min, y_max = min(points[:, 1]), max(points[:, 1])
                    points = [x_min, y_min, x_max, y_max]
                    point_list.append(points)
                    crop_img_1 = image.crop((x_min, y_min, x_max, y_max))
                    crop_img_2 = image.crop((x_min, y_min, x_max, y_max)).rotate(90, expand=True)
                    crop_img_3 = image.crop((x_min, y_min, x_max, y_max)).rotate(180, expand=True)
                    crop_img_4 = image.crop((x_min, y_min, x_max, y_max)).rotate(270, expand=True)
                    crop_img_list_1.append(crop_img_1)
                    crop_img_list_2.append(crop_img_2)
                    crop_img_list_3.append(crop_img_3)
                    crop_img_list_4.append(crop_img_4)
                
        for i in range (len(crop_img_list_1)):
            rec_result_1 = ocr.ocr(np.array(crop_img_list_1[i]), det=False, cls=False, rec=True)
            rec_result_2 = ocr.ocr(np.array(crop_img_list_2[i]), det=False, cls=False, rec=True)
            rec_result_3 = ocr.ocr(np.array(crop_img_list_3[i]), det=False, cls=False, rec=True)
            rec_result_4 = ocr.ocr(np.array(crop_img_list_4[i]), det=False, cls=False, rec=True)
            # rec_result = ocr.ocr(np.array(crop_img_list[i]))
            rec_result = max_num(rec_result_1,rec_result_2,rec_result_3,rec_result_4)
            # print(rec_result)
            rec_results_list.append({'text': rec_result, 'bbox':  point_list[i] })
        # final_result.append({'img_path': img_path,'rec_result':rec_results_list})
        final_result[img_path] = rec_results_list
    return final_result