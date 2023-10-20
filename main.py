import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from binarization import sauvola_threshold
from label import sequential_labeling


# 侵蝕
def dilate(img, max = 1):
    img_size = np.shape(img)
    img_delate = np.ones(img_size)
    for i in range (0,img_size[0]-1):
        for j in range (0, img_size[1]-1):            
            if(img[i-1, j-1]==255 or img[i-1,j] == max or img[i,j-1] == max or img[i+1,j] == max or img[i,j+1] == max or img[i+1,j+1] == max or img[i-1,j+1]== max or img[i+1,j-1]==max):
                img_delate[i, j] = max

    return img_delate

def boundary(img):
    s = np.argwhere(img == 0)[0]
    s = tuple(s)
    # 設定當前像素 c 和 4-鄰居 b
    c = s
    b = tuple[s[0], s[1] - 1]

    # 定義8-鄰居的順序
    neighbors_order = [(0, -1), (-1, -1), (-1, 0), (-1, 1),
                       (0, 1), (1, 1), (1, 0), (1, -1)]

    boundary_pixels = []  # 存儲邊界像素的座標

    while True:
        # 找到 8-鄰居中第一個屬於 S 的像素
        for i, (dx, dy) in enumerate(neighbors_order):
            ni = (c[0] + dx, c[1] + dy)
            if 0 <= ni[0] < img.shape[0] and 0 <= ni[1] < img.shape[1] and img[ni] != 0:
                break

        # 設置 c 和 b
        c = ni
        # b = [sum(x) for x in zip(ni, neighbors_order[(i - 1) % 8])]
        b = tuple((ni[0], ni[1] - 1))    
        neighbors_order = neighbors_order[i-1:] + neighbors_order[:i-1]

        # 將當前像素座標加入結果列表
        boundary_pixels.append(c)

        # 當 c 等於 s 時結束迴圈
        if c == s:
            break

    return boundary_pixels

    

def arrea_filter(img, min_area = 200, max_area = 5000):
    label_table = set(img.flatten())
    for label in label_table:
        img_label_filter = np.where(img == label, 1, 0)
        label_area = np.sum(img_label_filter)
        if(label_area<min_area)or(label_area>max_area):
            img = np.where(img == label, 0, img)
    return img
        




if __name__ == '__main__':

    #紀錄程式開始時間
    start_time = time.time()

    img = cv2.imread("sample/test2.bmp", cv2.IMREAD_GRAYSCALE)
    img_sauvola = sauvola_threshold(img.copy(), max =1, block_size = 30, R = 0.2) # 二值化矩陣

    # plt.imshow(img_sauvola, cmap='gray', vmin=0, vmax=1)
    # plt.title('sauvola')
    # plt.show()



    # NOTE 功能:侵蝕然後膨脹，但重要性未知，先使用內建函數跳過
    kernel = np.ones((2, 2), np.uint8)
    img_dilation = cv2.dilate(img_sauvola, kernel, iterations=1)  # 侵蝕
    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)  # 膨脹

    # label
    img_label = sequential_labeling(img_erosion.copy())
    unique_labels = set(img_label.flatten())
    print(unique_labels)
    print("區域數量:", len(unique_labels)-1)

    img_area_filter = arrea_filter(img_label, 200, 5000)
    unique_labels = set(img_area_filter.flatten())
    print("\n After area filter: \n")
    print(unique_labels)
    print("區域數量:", len(unique_labels)-1)



    # # 偵測邊界
    # boundary_pixel = boundary(img_label.copy()) 
    # result_image = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    # for pixel in boundary_pixel:
    #     result_image[pixel] = [0, 0, 255]  # 在 BGR 圖像中，紅色表示邊界
    # cv2.imshow('Boundary Image', result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    #計算程式運行時間
    end_time = time.time()
    print(f"Average Time: {end_time-start_time}s.")
    
    plt.figure(figsize=(15,15))
    plt.subplot(121)
    plt.imshow(img_sauvola, cmap='gray', vmin=0, vmax=1)
    plt.title('label')

    plt.subplot(122)
    plt.imshow(img_area_filter, cmap='gray', vmin=0, vmax=1)
    plt.title('area_filter')

    plt.show()    
