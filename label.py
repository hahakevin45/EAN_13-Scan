import numpy as np

def sequential_labeling(img):
    def find_lowest_label(equivalence, label):
        while equivalence[label] != label:
            label = equivalence[label]
        return label

    def update_equivalence(equivalent_labels, label1, label2):
        root1 = find_lowest_label(equivalent_labels, label1)
        root2 = find_lowest_label(equivalent_labels, label2)

        if root1 != root2:
            equivalent_labels[root1] = root2

    img_size = np.shape(img)
    labels = np.zeros_like(img, dtype=int)
    equivalence = list(range(img_size[0] * img_size[1]))

    current_label = 1

    # 第一遍掃描
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            if img[i, j] == 1:
                upper_label = labels[i - 1, j] if i > 0 else 0
                left_label = labels[i, j - 1] if j > 0 else 0

                if upper_label == 0 and left_label == 0:
                    labels[i, j] = current_label
                    current_label += 1
                elif upper_label == 0:
                    labels[i, j] = left_label
                elif left_label == 0:
                    labels[i, j] = upper_label
                elif upper_label == left_label:
                    labels[i, j] = upper_label
                else:
                    labels[i, j] = upper_label
                    update_equivalence(equivalence, upper_label, left_label)

    # 第二遍掃描
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            if img[i, j] == 1:
                labels[i, j] = find_lowest_label(equivalence, labels[i, j])
                
    # 使用 set 獲取唯一標籤的數量
    unique_labels = set(labels.flatten())

    # 不包括背景標籤 0
    region_count = len(unique_labels) - 1

    print("區域數量:", region_count)    

    return labels

if __name__ == "__main__":
    # 示例用法：
    binary_image = np.array([[0, 1, 0, 0, 1],
                            [1, 1, 0, 1, 0],
                            [0, 0, 1, 1, 0],
                            [1, 0, 0, 0, 1]])

    labeled_image = sequential_labeling(binary_image)
    print(labeled_image)
    
