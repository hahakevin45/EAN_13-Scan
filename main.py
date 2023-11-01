import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from binarization import sauvola_threshold
from label import sequential_labeling
from filter import arrea_filter
from feature import Label
from feature import least_square_method
from decode import decode

if __name__ == '__main__':

    # 紀錄程式開始時間
    start_time = time.time()

    # 使用cv2讀取照片
    img = cv2.imread("sample/test1.bmp", cv2.IMREAD_GRAYSCALE)
    size_of_image = np.shape(img)
    # print(f"size of image {size_of_image}")

    # Use sauvla theresshold to binarize image
    img_sauvola = sauvola_threshold(img.copy(), max=1, block_size=30, R=0.2)

    # label
    img_label = sequential_labeling(img_sauvola.copy())
    unique_labels = set(img_label.flatten())
    print("Afer label")
    print(unique_labels)
    print("區域數量:", len(unique_labels)-1)

    # area_filter
    img_area_filter = arrea_filter(img_label, size_of_image[0]*size_of_image[1]/2400, size_of_image[0]*size_of_image[1]/30)
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
    # for label in label_list:
    #     if label._perimeter*label._perimeter/label._area < 60:
    #         label_list.remove(label)
    #     print(f"Label value {label.value}, perimeter {label._perimeter}")

    # 檢查mass 是否通過 區塊本身
    # for label in label_list:
    #     neighbors_order = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]
    #     std_mass = [round(label._mass[0]),round(label._mass[1])]
    #     in_mass = False
    #     for neighbors in neighbors_order:
    #         std_mass = [std_mass[0] + neighbors[0], std_mass[1]+neighbors[1]]
    #         if std_mass[0]>size_of_image[0]-1:
    #             std_mass[0]==[size_of_image[0]-1,std_mass[1]]                     
    #         if std_mass[1]>size_of_image[1]-1:
    #             std_mass = [std_mass[0],size_of_image[1]-1]
    #         if label._pixels[std_mass[0],std_mass[1]]==1:
    #             in_mass = True
    #     if in_mass != True:
    #         label_list.remove(label)


    # claculate line
    line = least_square_method(list(label._mass for label in label_list))
    print(f"Line: y = {line[0]} x + {line[1]}")


    for label in label_list:
        label.found_distance(line)
        # print(f"label: {label.value} Distance: {label.distance} Mass: {label._mass}")
        if label.distance > 0.6:
            label_list.remove(label)

    # 顯示包含在 label_list 的圖片
    isperimetric_incequality_unique = set()
    for label in label_list:
        isperimetric_incequality_unique.add(label.value)
    result = np.isin(img_area_filter, list(isperimetric_incequality_unique)) # 使用 np.isin 產生布林數組，True 表示在 unique_labels 中，False 表示不在
    result = result.astype(int) * img_area_filter # 將 True 替換為對應的值，False 替換為 0
    unique_labels = set(result.flatten())
    print("\n After mass filter: \n")
    unique_labels.remove(0)
    print(unique_labels)
    print("區域數量:", len(unique_labels))

    # 偵測EAN-13 cod3
    result = np.where(result > 1, 1, 0)
    ean13, is_valid, thresh = decode(result)

    # 計算程式運行時間
    end_time = time.time()
    print(f"Average Time: {end_time-start_time}s.")
    

    # show image 1
    plt.figure(figsize=(15, 15))
    plt.subplot(121)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title('Original image')

    # show image 2
    plt.subplot(122)
    plt.imshow(result, cmap='gray', vmin=0, vmax=1)
    plt.text(30, 50, ean13, {'fontsize':18},bbox={
    'boxstyle':'round',
    'facecolor':'#F5EEC8',
    'edgecolor':'#555843',
    'pad':0.5,
    'linewidth':3})
    plt.title('rusult')

    plt.show()
