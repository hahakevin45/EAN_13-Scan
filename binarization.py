import numpy as np


# 二值化矩陣
def binarization(img, max=1, C=0):
    T = np.mean(img) # 設 T 為平均值
    img = np.where(img > (T - C), max, 0)
    return img


# 使用Adaptive Threshold 二值化矩陣
def adaptive_threshold(img, max=1, block_size=20, C=0):
    img_size = np.shape(img)
    for i in range(img_size[0]//block_size):
        for j in range(img_size[1]//block_size):           
            temp_block = img[(i)*block_size:(i+1)*block_size, (j)*block_size:(j+1)*block_size] # 提取 block size 大小的範圍
            temp_block = binarization(temp_block, max, C) # block size 範圍內二值化
            img[(i)*block_size:(i+1)*block_size, (j)*block_size:(j+1)*block_size] = temp_block
    return img

# 計算Sauvola閾值
def sauvola_threshold(img, max = 1, block_size=30, sauvola_R=0.5):
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
            threshold = mean * (1 + sauvola_R * ((std_dev / 128) - 1))
            
            # 根據閾值進行二值化
            threshold_img[i:i+block_size, j:j+block_size] = (block < threshold) * max

    return threshold_img