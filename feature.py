import numpy as np


# 尋找重心

def found_mass(img, value, area) -> list:
    img = np.where(img == value, 1, 0)

    # 計算重心座標
    mass_i = np.sum(np.sum(img, axis=0) * np.arange(img.shape[1]))
    mass_j = np.sum(np.sum(img, axis=1) * np.arange(img.shape[0]))

    return [mass_i / area, mass_j / area]

# 尋找邊界


def found_boundary(img, value) -> list:
    img = np.where(img == value, 1, 0)
    s = np.argwhere(img == 1)[0]
    s = tuple(s)
    # 設定當前像素 c 和 4-鄰居 b
    c = s
    # b = tuple[s[0], s[1] - 1]

    # 定義8-neighbor的順序
    neighbors_order = [(0, -1), (-1, -1), (-1, 0), (-1, 1),
                       (0, 1), (1, 1), (1, 0), (1, -1)]

    boundary_pixels = [s]  # 存儲邊界像素的座標

    while True:
        # 找到 8-鄰居中第一個屬於 S 的像素
        for i, (dx, dy) in enumerate(neighbors_order):
            ni = (c[0] + dx, c[1] + dy)
            if 0 <= ni[0] < img.shape[0] and 0 <= ni[1] < img.shape[1] and img[ni] != 0:
                break

        # 設置 c 和 b
        c = ni
        # b = tuple((ni[0], ni[1] - 1))
        neighbors_order = neighbors_order[i-1:] + neighbors_order[:i-1]

        # 當 c 已經存在於 boundary_pixels 時結束迴圈
        if c in boundary_pixels:
            break

        # 將當前像素座標加入結果列表
        boundary_pixels.append(c)

        # 當 c 等於 s 時結束迴圈
        # if c == s :
        #     break

    return boundary_pixels

# 求重心近似直線


def least_square_method(mass_list):
    mean_x = sum(item[0] for item in mass_list)/len(mass_list)
    mean_y = sum(item[1] for item in mass_list)/len(mass_list)
    a = sum((item[0]-mean_x)*(item[1]-mean_y) for item in mass_list) / \
        sum((item[0]-mean_x)**2 for item in mass_list)

    b = mean_y - a*mean_x
    return ([a, b])


class Label:
    def __init__(self, value, img) -> None:
        self.value = value
        self.img = img

        self._pixels = np.where(self.img == self.value,
                                1, 0)  # 將label獨立出來並設成 1
        self._area = np.sum(self._pixels)  # 計算像素個數

        self._mass = found_mass(self.img, self.value, self._area)

        self._boundary_pixels = found_boundary(self.img, self.value)
        self._perimeter = len(self._boundary_pixels)

    def found_distance(self, line):
        self.distance = abs(
            line[0]*self._mass[0]-self._mass[1]+line[1])/((line[0]**2+line[1]**2)**0.5)
