import cv2
import numpy as np


# 二值化矩陣
def binarization(img, max=255, C=0):
    T = np.mean(img) # 計算平均值
    img = np.where(img > (T - C), max, 0)
    return img

# 使用Adaptive Threshold 二值化矩陣
def adaptive_threshold(img, max=255, block_size=20, C=0):
    img_size = np.shape(img)
    for i in range(img_size[0]//block_size):
        for j in range(img_size[1]//block_size):
            temp_block = img[(i)*block_size:(i+1)*block_size, (j)*block_size:(j+1)*block_size]
            temp_block = binarization(temp_block, max, C)
            img[(i)*block_size:(i+1)*block_size, (j)*block_size:(j+1)*block_size] = temp_block
    return img

# 計算Sauvola閾值
def sauvola_threshold(img, block_size=20, R=0.5):
    img_size = np.shape(img)
    threshold_img = np.zeros_like(img, dtype=np.uint8)

    for i in range(0, img_size[0], block_size):
        for j in range(0, img_size[1], block_size):
            # 提取區域
            block = img[i:i+block_size, j:j+block_size]
            
            # 計算平均值和標準差
            mean = np.mean(block)
            std_dev = np.std(block)
            
            # 計算Sauvola閾值
            threshold = mean * (1 + R * ((std_dev / 128) - 1))
            
            # 根據閾值進行二值化
            threshold_img[i:i+block_size, j:j+block_size] = (block > threshold) * 255

    return threshold_img

# 侵蝕
def dilate(img, max = 255):
    img_size = np.shape(img)
    img_delate = np.ones(img_size)
    for i in range (0,img_size[0]-1):
        for j in range (0, img_size[1]-1):            
            if(img[i-1, j-1]==255 or img[i-1,j] == max or img[i,j-1] == max or img[i+1,j] == max or img[i,j+1] == max or img[i+1,j+1] == max or img[i-1,j+1]== max or img[i+1,j-1]==max):
                img_delate[i, j] = max

    return img_delate


    
if __name__ == '__main__':

    img = cv2.imread("sample/test1.bmp", cv2.IMREAD_GRAYSCALE)
    # img_bin = adaptive_threshold(img.copy(), 255, 20, 10) # 二值化圖片
    img_sauvola = sauvola_threshold(img.copy(), 16, 0.2)

    # # NOTE 功能:侵蝕然後膨脹，但重要性未知，先使用內建函數跳過
    # kernel = np.ones((2, 2), np.uint8)
    # dilation = cv2.dilate(img_sauvola, kernel, iterations=1)  # 侵蝕
    # erosion = cv2.erode(dilation, kernel, iterations=1)  # 膨脹
    # cv2.imshow('erosion', erosion)


    cv2.imshow("sauvola",img_sauvola)
    # dilation = dilate(img_sauvola.copy())
    # cv2.imshow("dilation",dilation)
    cv2.waitKey()
    cv2.destroyAllWindows()
