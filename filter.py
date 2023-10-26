import numpy as np

from feature import found_boundary

def arrea_filter(img, min_area=200, max_area=5000):
    "過濾面積大於 max_area 和小於 min_area 面積的label"
    label_table = set(img.flatten())
    for label in label_table:
        img_label_filter = np.where(img == label, 1, 0)  # 將label獨立出來並設成 1
        label_area = np.sum(img_label_filter)  # 計算像素個數

        # 過濾過大和過小面積的label
        if (label_area < min_area) or (label_area > max_area):
            img = np.where(img == label, 0, img)

    return img

def isoperimetric_inequality(img, label): 
    img = np.where(img == label, 1, 0)  # 將label獨立出來並設成 1  
    perimeter = len(found_boundary(img))
    area = np.sum(img) / label 
    return perimeter*perimeter/area

def least_square_method(mass_list):
    mean_x = sum(mass_list[0])/len(mass_list)
    mean_y = sum(mass_list[1])/len(mass_list)
    a = 
    b = mean_y - a*mean_x
    return([a, b])