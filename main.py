import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from binarization import sauvola_threshold
from label import sequential_labeling
from filter import arrea_filter
from filter import isoperimetric_inequality


if __name__ == '__main__':

    # 紀錄程式開始時間
    start_time = time.time()

    # 使用cv2讀取照片
    img = cv2.imread("sample/test2.bmp", cv2.IMREAD_GRAYSCALE)

    # 使用 sauvla theresshold 二值化矩陣
    img_sauvola = sauvola_threshold(img.copy(), max=1, block_size=30, R=0.2)

    # # HACK 功能:侵蝕然後膨脹，但重要性未知，先使用內建函數跳過
    # kernel = np.ones((2, 2), np.uint8)
    # img_dilation = cv2.dilate(img_sauvola, kernel, iterations=1)  # 侵蝕
    # img_erosion = cv2.erode(img_dilation, kernel, iterations=1)  # 膨脹

    # label
    img_label = sequential_labeling(img_sauvola.copy())
    unique_labels = set(img_label.flatten())
    print(unique_labels)
    print("區域數量:", len(unique_labels)-1)

    # area_filter
    img_area_filter = arrea_filter(img_label, 200, 4000)
    unique_labels = set(img_area_filter.flatten())
    unique_labels.remove(0)
    print("\n After area filter: \n")
    print(unique_labels)
    print("區域數量:", len(unique_labels))

    # img_in_filter = boundary(img_area_filter)
    for label in unique_labels:
        label_phase = dict(label, isoperimetric_inequality(img_area_filter, label))

    for label in unique_labels:
        if label_phase < 10000:
            img_isoperimetric_inequality_filter = np.where(img_area_filter == label, 0, img_area_filter)

    unique_labels = set(img_isoperimetric_inequality_filter.flatten())
    unique_labels.remove(0)
    print("\n After isoperimetric inquality filter: \n")
    print(unique_labels)
    print("區域數量:", len(unique_labels))
   

    # 計算程式運行時間
    end_time = time.time()
    print(f"Average Time: {end_time-start_time}s.")

    # show image
    plt.figure(figsize=(15, 15))
    plt.subplot(122)
    plt.imshow(img_isoperimetric_inequality_filter, cmap='gray', vmin=0, vmax=1)
    plt.title('Isoperimetric Inequality')

    # show image
    plt.subplot(121)
    plt.imshow(img_area_filter, cmap='gray', vmin=0, vmax=1)
    plt.title('area_filter')

    plt.show()
