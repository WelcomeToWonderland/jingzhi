import sys
import time
import os

def component_det(img_path):

    outname = str(time.time())
    results_dir = os.path.join("./component_results", outname)
    cat_int_to_name_dict = {0: 'component'}

    script_path = './yoltv5/yolov5/detect.py'
    weights_file = './yoltv5/yolov5/weigths/component2/weights/best.pt'
    yolo_cmd = 'python {} --weights {} --source {} --img {} --conf {} ' \
                '--name {} --project {} --nosave --save-txt --save-conf'.format(\
                script_path, weights_file, img_path, 1280, 0.6, outname, "./component_results")
    print("\nyolo_cmd:", yolo_cmd)
    os.system(yolo_cmd)

    base_name = os.path.basename(img_path)
    file_name, _ = os.path.splitext(base_name)
    pred_dir = os.path.join(results_dir + '', 'labels', file_name + '.txt')
    bboxes = []
    with open(pred_dir, 'r') as file:
        for line in file:
            parts = line.strip().split()
            bbox = list(map(float, parts[1:5]))  # Extract x1, y1, x2, y2
            bboxes.append(bbox)
    return bboxes

bboxes = component_det('./data/component/1.png')
print(bboxes)