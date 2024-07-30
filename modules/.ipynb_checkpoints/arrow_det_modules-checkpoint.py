import sys
import time
import os
sys.path.append("./yoltv5")
import tile_ims_labels
import post_process

### 输入构件图像路径，返回检测箭头的bbox列表
def arrow_det(img_path):

    outname = str(time.time())
    results_dir = os.path.join("./slice_arrow_results", outname)
    cat_int_to_name_dict = {0: 'arrow'}

    outdir =  os.path.join("./slice", outname)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print("outdir_slice_ims:", outdir)
       
        im_name = os.path.basename(img_path)
        out_name = im_name.split('.')[0]
        tile_ims_labels.slice_im_plus_boxes(img_path, out_name, outdir, sliceHeight=320, sliceWidth=320, slice_sep='__')

    script_path = './yoltv5/yolov5/detect.py'
    weights_file = './yoltv5/yolov5/weigths/3_3_arrow/weights/best.pt'
    yolt_cmd = 'python {} --weights {} --source {} --img {} --conf {} ' \
                '--name {} --project {} --nosave --save-txt --save-conf'.format(\
                script_path, weights_file, outdir, 320, 0.6, outname, "./slice_arrow_results")
    print("\nyolt_cmd:", yolt_cmd)
    os.system(yolt_cmd)
            

    pred_dir = os.path.join(results_dir + '', 'labels')
    print("post-proccessing:", outname)
    result = None

    result = post_process.execute(
        pred_dir=pred_dir,
        cat_int_to_name_dict=cat_int_to_name_dict,
        slice_size=320,
        sep='__',
        detection_thresh=0.6)

    if result is None:
        return []
    
    arrow_bboxes = []
    for arrow_info in result.iloc:
        x1 = int(arrow_info['Xmin_Glob'])
        y1 = int(arrow_info['Ymin_Glob'])
        x2 = int(arrow_info['Xmax_Glob'])
        y2 = int(arrow_info['Ymax_Glob'])
        arrow_bboxes.append([x1, y1, x2, y2])

    return arrow_bboxes

### 输入构件图像路径列表，返回字典
def pipeline_arrow_det(img_path_list):
    result = {}
    for img_path in img_path_list:
        result[img_path] = arrow_det(img_path)
    return result

### test

# import cv2

# log_root = './logs'
# logname = '构件箭头检测demo展示_精智'

# img_dir = './data/精智demo展示案例备选二'
# img_path_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
# if not os.path.exists(os.path.join(log_root, logname)):
#     os.makedirs(os.path.join(log_root, logname))

# arrow_dic = pipeline_arrow_det(img_path_list)
# for img_path, arrow_bboxes in arrow_dic.items():
#     baseneame = os.path.basename(img_path)
#     prefix, suffix = os.path.splitext(baseneame)
#     image = cv2.imread(img_path)
#     for arrow in arrow_bboxes:
#         x1, y1, x2, y2 = arrow
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     # 保存图片
#     output_path = os.path.join(log_root, logname, prefix + '_arrow_det' + suffix)
#     cv2.imwrite(output_path, image)