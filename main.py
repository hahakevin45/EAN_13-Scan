import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from binarization import sauvola_threshold
from label import sequential_labeling
from filter import arrea_filter
from feature import Label

if __name__ == '__main__':

    # 紀錄程式開始時間
    start_time = time.time()

    # 使用cv2讀取照片
    img = cv2.imread("sample/test1.bmp", cv2.IMREAD_GRAYSCALE)
    size_of_image = np.shape(img)

    print(size_of_image)
    # Use sauvla theresshold to binarize image
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
    img_area_filter = arrea_filter(img_label, size_of_image[0]*size_of_image[1]/2400, size_of_image[0]*size_of_image[1]/80)
    unique_labels = set(img_area_filter.flatten())
    unique_labels.remove(0)
    print("\n After area filter: \n")
    print(unique_labels)
    print("區域數量:", len(unique_labels))

    
    # label 物件化
    label_list = []
    for label in unique_labels:
        label_list.append(Label(label, img_area_filter))

    # 圓周不等式 filter
    for label in label_list:
        if label._perimeter*label._perimeter/label._area < 60:
            label_list.remove(label)
        # print(f"Label value {label.value}, perimeter {label._perimeter}")


    # # 單獨測試 label     
    # label_test = Label(953, img_area_filter);
    # print(label_test.value)
    # print(label_test._boundary_pixels)
    # print(label_test._perimeter)

    # 顯示包含在 label_list 的圖片檔
    isperimetric_incequality_unique = set()
    for label in label_list:
        isperimetric_incequality_unique.add(label.value)
    # 使用 np.isin 生成布尔数组，True 表示在 unique_labels 中，False 表示不在
    result = np.isin(img_area_filter, list(isperimetric_incequality_unique))

    # 将布尔数组中的 True 替换为对应的值，False 替换为 0
    result = result.astype(int) * img_area_filter
        
    unique_labels = set(result.flatten())
    print("\n After 圓周不等式 filter: \n")
    unique_labels.remove(0)
    print(unique_labels)
    print("區域數量:", len(unique_labels))

    # 計算程式運行時間
    end_time = time.time()
    print(f"Average Time: {end_time-start_time}s.")
    

    # show image 2
    plt.figure(figsize=(15, 15))
    plt.subplot(122)
    plt.imshow(result, cmap='gray', vmin=0, vmax=10)
    plt.title('img')

    # show image 1
    plt.subplot(121)
    plt.imshow(img_area_filter, cmap='gray', vmin=0, vmax=1)
    plt.title('img_area_filter')

    plt.show()
