import cv2
import numpy as np

img = cv2.imread("sample/test1.bmp", cv2.IMREAD_GRAYSCALE)
img_size = np.shape(img)
print("img_size: ", img_size)

img_average_value = np.sum(img,axis = (0,1))/(img_size[0]*img_size[1])
# print("img sum: ", img_average_value)

#TODO 這裡要自己實現，不能用函數
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 5)

# for i in range(img_size[0]):
#     for j in range(img_size[1]):
#         if img[i][j] > img_average_value:
#             img[i][j] = 255
#         elif img[i][j] <= img_average_value:
#             img[i][j] = 0


cv2.imshow("fig1", img)
cv2.waitKey()