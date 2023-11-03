import cv2
import numpy as np
import time

from binarization import sauvola_threshold
from label import sequential_labeling
from filter import arrea_filter
from feature import Label, least_square_method
from decode import decode
from show_image import show_rusult_image, show_gray_image, label_list_to_img


def process(img, max_value,sauvola_block_size , sauvola_R ):

    size_of_image = np.shape(img)

    # Use sauvla theresshold to binarize image
    img_sauvola = sauvola_threshold(img.copy(), max_value, sauvola_block_size, sauvola_R)

    # label
    img_label = sequential_labeling(img_sauvola.copy())
    unique_labels = set(img_label.flatten())
    unique_labels.remove(0)

    # area_filter
    img_area_filter = arrea_filter(img_label, size_of_image[0]*size_of_image[1]/2400,  size_of_image[0]*size_of_image[1]/30)
    unique_labels = set(img_area_filter.flatten())

    
    # label 物件化
    label_list = []
    for label in unique_labels:
        label_list.append(Label(label,img_label))

    # area filter on label list
    # for label in label_list:
        # print(label._area)
        # if (label._area < size_of_image[0]*size_of_image[1]/2400) or (label._area > 0):
            # label_list.remove(label)


    # claculate line
    line = least_square_method(list(label._mass for label in label_list))
    print(f"Line: y = {line[0]} x + {line[1]}")

    #過濾離線過遠的區塊
    for label in label_list:
        label.found_distance(line)
        # print(f"label: {label.value} Distance: {label.distance} Mass: {label._mass}")
        if label.distance > 0.6:
            label_list.remove(label)
    
    # 將 label_list 剩下的 label 顯示在rusult
    result = label_list_to_img(label_list, img_area_filter)

    # 偵測EAN-13 cod3
    ean13, is_valid, thresh = decode(result)

    return ean13, is_valid, thresh




if __name__ == '__main__':

    # 紀錄程式開始時間
    start_time = time.time()

    # 使用cv2讀取照片
    img = cv2.imread("\sample\data1_image\image_1.bmp", cv2.IMREAD_GRAYSCALE)

    #辨識圖片的EAN-13條碼
    ean13, is_valid, thresh = process(img, max_value=1, sauvola_block_size=30,sauvola_R= 0.2)


    try_way = "First try"

    if is_valid == None:
        try_way = "block_size = 90 TRY"
        ean13, is_valid, thresh = process(img, max_value=1, sauvola_block_size=90,sauvola_R= 0.2)
    if is_valid == None:
        try_way = "k=1 TRY"
        img_rotation = np.rot90(img, k=1)
        ean13, is_valid, thresh = process(img_rotation, max_value=1, sauvola_block_size=60,sauvola_R= 0.2)
    
    if is_valid == None:
        try_way = "k=-1 TRY"
        img_rotation = np.rot90(img, k=-1)
        ean13, is_valid, thresh = process(img_rotation, max_value=1, sauvola_block_size=60,sauvola_R= 0.2)



    print(f"--------------------finish at {try_way}-----------------------------")

    # 計算程式運行時間
    end_time = time.time()
    
    print(f"Average Time: {end_time-start_time}s.")
    show_gray_image(img)
    show_rusult_image(thresh, ean13)

 