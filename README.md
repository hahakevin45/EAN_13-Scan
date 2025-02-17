# 📌 EAN-13 Barcode Recognition (無依賴第三方條碼庫)



## 🚀 專案簡介
本專案為 **機器視覺課程期末專案**，目標是 **不使用任何現成條碼辨識函式庫（如 OpenCV、ZBar）**，純手寫演算法來進行 EAN-13 條碼辨識。

本專案包含：
- **影像預處理**：圖片正規化、二值化、尺寸與旋轉調整
- **條碼特徵提取**：標籤處理、重心過濾、圓周不等式過濾
- **條碼解碼演算法**：基於 `similar edge distance algorithm` 來解析條碼
- **效能測試**：測試 215 張圖片，正確率達 **66.9%**，平均運行時間 **2.128 秒**

---

## 📷 成果展示

<img src="https://github.com/user-attachments/assets/f7262fed-ba1f-4e04-bedd-4294daa818d3" alt="條碼辨識結果" width="600">

<img src="https://github.com/user-attachments/assets/8705b65b-4721-4290-83af-8715e2ef71cf" alt="條碼辨識結果" width="600">

---

## 📖 主要技術
### 📌 **影像前處理**
1. **圖片正規化**：將圖片長邊縮放至 `600 px`
2. **旋轉補償**：嘗試 ±45°、±90° 旋轉條碼
3. **二值化**：使用 `Sauvola threshold` 方法進行二值化
4. **尺寸過濾**：使用 `size filter` 過濾非條碼區域

### 🔍 **條碼特徵提取**
- **標籤與尺寸過濾**
- **重心過濾**
- **圓周不等式過濾**

### 🎯 **條碼解碼演算法**
- **使用 `similar edge distance algorithm` 解析條碼**
- **比對 EAN-13 標準來解碼數字**
- **自動計算 `Check Digit` 進行驗證**

---

## 📊 測試結果
- 📌 **總測試圖片**：`215 張`
- ✅ **成功辨識圖片**：`144 張`
- 🎯 **正確率**：`66.9%`
- ⏱ **平均運行時間**：
  - **正確圖片**：`2.128 秒`
  - **全部圖片**：`8.412 秒`

---

## 📦 安裝與執行
### **1️⃣ 環境需求**
請確保你已安裝 **Python 3.x**，並安裝以下套件：
```sh
pip install -r requirements.txt
```
📌 單張圖片測試
```shell
python main.py --input sample/test1.bmp
```
📌 批次處理整個資料夾
```shell
python main.py --input_folder sample/
```

## Reference
1. “Locating and Decoding EAN-13 Barcodes using Python and OpenCV,” Dynamsoft Developers Blog. Accessed: Nov.07, 2023. [Online]. Available: https://www.dynamsoft.com/codepool/locating-and-decoding-ean13-python-opencv.html
1. J. Sauvola and M. Pietikäinen, “Adaptive document image binarization,” Pattern Recognition, vol. 33, no. 2, pp. 225–236, Feb. 2000, doi: 10.1016/S0031-3203(99)00055-2.
1. W. Niblack, An introduction to digital image processing. Englewood Cliffs: Prentice-Hall, 1986.
1. Neural Image Restoration For Decoding 1-D Barcodes Using Common Camera Phones Alessandro Zamberletti, Ignazio Gallo, Moreno Carullo and Elisabetta Binaghi Computer Vision, Imaging and Computer Graphics. Theory and Applications, Springer Berlin Heidelberg, 2011
