import cv2
import numpy as np
import time

from binarization import sauvola_threshold
from label import sequential_labeling
from filter import arrea_filter
from feature import Label, least_square_method
from decode import decode
from show_image import show_rusult_image, show_gray_image, label_list_to_img


def process(img, max_value,sauvola_block_size , sauvola_R, distance_filter ,min_scale = 2400, max_scale = 30):

    size_of_image = np.shape(img)

    # Use sauvla theresshold to binarize image
    img_sauvola = sauvola_threshold(img.copy(), max_value, sauvola_block_size, sauvola_R)

    # label
    img_label = sequential_labeling(img_sauvola.copy())
    unique_labels = set(img_label.flatten())
    unique_labels.remove(0)

    # area_filter
    img_area_filter = arrea_filter(img_label, size_of_image[0]*size_of_image[1]/min_scale,  size_of_image[0]*size_of_image[1]/max_scale)
    unique_labels = set(img_area_filter.flatten())

    
    # label 物件化
    label_list = []
    for label in unique_labels:
        label_list.append(Label(label,img_label))

    
    # claculate line
    line = least_square_method(list(label._mass for label in label_list))
    print(f"Line: y = {line[0]} x + {line[1]}")

    #過濾離線過遠的區塊
    for label in label_list:
        # print(f"Len of list: {len(label_list)}")
        label.found_distance(line)
        # print(f"label: {label.value} Distance: {label.distance} Mass: {label._mass}")
    label_list = [x for x in label_list if x.distance <distance_filter]

    print("After mass filter")
    # print(f"label list {label_list}")

    # 將 label_list 剩下的 label 顯示在rusult
    result = label_list_to_img(label_list, img_area_filter)

    # 偵測EAN-13 cod3
    ean13, is_valid, thresh = decode(result)


    return ean13, is_valid, thresh

def rotate_bound_white_bg(img, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = (img.shape[0], img.shape[1])
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    matrix = cv2.getRotationMatrix2D((cX, cY), -angle, 0.75)
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
 
    # compute the new bounding dimensions of the image
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    matrix[0, 2] += (new_w / 2) - cX
    matrix[1, 2] += (new_h / 2) - cY
 
    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    return cv2.warpAffine(img, matrix, (new_w, new_h),borderValue=(255,255,255))
    # borderValue 缺省，默认是黑色（0, 0 , 0）
    # return cv2.warpAffine(image, M, (nW, nH))
 


if __name__ == '__main__':

    # 紀錄程式開始時間
    start_time = time.time()

    # 使用cv2讀取照片
    img_re = cv2.imread("sample/data1_image/image_10.bmp", cv2.IMREAD_GRAYSCALE)
    # img_re = cv2.imread("sample/test6.bmp", cv2.IMREAD_GRAYSCALE)
    size_of_image = np.shape(img_re)

    # 正規化圖片大小
    scale =600/max(size_of_image[0],size_of_image[1])
    img = cv2.resize(img_re, (int(size_of_image[1]*scale), int(size_of_image[0]*scale)), interpolation=cv2.INTER_AREA)

    #辨識圖片的EAN-13條碼
    ean13, is_valid, thresh = process(img, max_value=1, sauvola_block_size = 30 , sauvola_R= 0.2, distance_filter=0.6)
    try_way = "First try"


    if is_valid == None or is_valid == False:
        try_way = "block_size = 90 TRY"
        ean13, is_valid, thresh = process(img, max_value=1, sauvola_block_size=90,sauvola_R= 0.2, distance_filter=0.6)

    if is_valid == None or is_valid == False:
        try_way == "rotate -45"
        img_rotation = rotate_bound_white_bg(img, -45)
        ean13, is_valid, thresh = process(img_rotation, max_value=1, sauvola_block_size=90,sauvola_R= 0.2, distance_filter=0.3)
     
    if is_valid == None or is_valid == False:
        try_way == "rotate -45 and small decode"
        img_rotation = rotate_bound_white_bg(img, -45)
        ean13, is_valid, thresh = process(img_rotation, max_value=1, sauvola_block_size=120,sauvola_R= 0.2, distance_filter=10, min_scale=20000, max_scale=70)
     
    if is_valid == None or is_valid == False:
        try_way == "rotate 45"
        img_rotation = rotate_bound_white_bg(img, 45)
        ean13, is_valid, thresh = process(img_rotation, max_value=1, sauvola_block_size=90,sauvola_R= 0.2, distance_filter=0.6)

    if is_valid == None or is_valid == False:
        try_way = "small decode"
        ean13, is_valid, thresh = process(img, max_value=1, sauvola_block_size=100,sauvola_R= 0.2, distance_filter=6, min_scale=30000,max_scale=70)

   
    # 旋轉圖形
    if is_valid == None or is_valid == False:
        try_way = "k=1 TRY"
        img_rotation = np.rot90(img, k=1)
        ean13, is_valid, thresh = process(img_rotation, max_value=1, sauvola_block_size=60,sauvola_R= 0.2, distance_filter=0.6)
    
    if is_valid == None or is_valid == False:
        try_way = "k=-1 TRY"
        img_rotation = np.rot90(img, k=-1)
        ean13, is_valid, thresh = process(img_rotation, max_value=1, sauvola_block_size=60,sauvola_R= 0.2, distance_filter= 0.6)
    if is_valid == None or is_valid == False:
        try_way = "k=2 TRY"
        img_rotation = np.rot90(img, k=2)
        ean13, is_valid, thresh = process(img_rotation, max_value=1, sauvola_block_size=60,sauvola_R= 0.2, distance_filter=0.6)
    
    
    print(f"--------------------End at ({try_way})-----------------------------")

    # 計算程式運行時間
    end_time = time.time()
    
    print(f"Average Time: {end_time-start_time}s.")
    show_gray_image(img_re)
    show_rusult_image(thresh, ean13)

 