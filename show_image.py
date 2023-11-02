
import matplotlib.pyplot as plt
import numpy as np


def show_gray_image(img):
    'show gray image'
    plt.figure(figsize=(15, 15))
    plt.subplot(121)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title('Original image')


def show_rusult_image(result, ean13):
    'show binarization image.'
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


def label_list_to_img(label_list, img):
    '顯示包含在 label_list 的圖片'
    isperimetric_incequality_unique = set()
    for label in label_list:
        isperimetric_incequality_unique.add(label.value)
    result = np.isin(img, list(isperimetric_incequality_unique)) # 使用 np.isin 產生布林數組，True 表示在 unique_labels 中，False 表示不在
    result = result.astype(int) *img# 將 True 替換為對應的值，False 替換為 0

    result = np.where(result > 1, 1, 0)
    return result

