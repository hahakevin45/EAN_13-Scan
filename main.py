import cv2
import numpy as np




#計算傳入的矩陣平均數值
def img_average (img):
    img_size = np.shape(img)
    img_average_value = np.sum(img,axis = (0,1))/(img_size[0]*img_size[1])
    return img_average_value


#二值化矩陣
def binarization(img, max = 255, C=0):
    img_size = np.shape(img)
    img_average_value = img_average(img)    
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            if img[i][j] > (img_average_value-C):
                img[i][j] = max
            elif img[i][j] <= (img_average_value-C):
                img[i][j] = 0
    return img;


#使用Adaptive Threshold 二值化矩陣
def adaptive_threshold(img, max = 255, block_size = 21, C = 0):
    img_size = np.shape(img)

    for i in  range(img_size[0]//block_size):
        for j in  range(img_size[1]//block_size):
            temp_block = np.ones((block_size, block_size))
            temp_block = img[(i)*block_size:(i+1)*block_size, (j)*block_size:(j+1)*block_size]
            temp_block = binarization(temp_block, max, C)   
    return img;




if __name__ == '__main__':
    img = cv2.imread("sample/test1.bmp", cv2.IMREAD_GRAYSCALE)
    img_bin = adaptive_threshold(img, 255, 19, 15)
    
    cv2.imshow("fig1", img_bin)
    cv2.waitKey() 
    cv2.destroyAllWindows()


