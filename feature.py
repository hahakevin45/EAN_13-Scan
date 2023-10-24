import numpy as np


# 尋找重心
def fonund_mass(img) -> list:
    img = np.where(img > 1, 1, 0)
    mass_i = np.mean(img, axis=0)
    mass_j = np.mean(img, axis=1)
    return [mass_i, mass_j]


# 尋找邊界
def found_boundary(img) -> list:
    s = np.argwhere(img == 0)[0]
    s = tuple(s)
    # 設定當前像素 c 和 4-鄰居 b
    c = s
    # b = tuple[s[0], s[1] - 1]

    # 定義8-neighbor的順序
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
        # b = tuple((ni[0], ni[1] - 1))
        neighbors_order = neighbors_order[i-1:] + neighbors_order[:i-1]

        # 將當前像素座標加入結果列表
        boundary_pixels.append(c)

        # 當 c 等於 s 時結束迴圈
        if c == s:
            break

    return boundary_pixels


class Label:
    def __init__(self, value, img) -> None:
        self.value = value
        self.img = img
        
        self._pixels = np.where(self.img == self.value, 1, 0)  # 將label獨立出來並設成 1
        self._ares = np.sum(self._pixels)  # 計算像素個數

        self._mass = fonund_mass(self.img)
        self._boundary_pixels = found_boundary(self.img)

        self._perimeter = len(self._boundary_pixels)

