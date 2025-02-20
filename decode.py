

# 解碼函式，輸入二值化的圖片，進行EAN-13條碼解碼
def decode(img):
    ean13 = None
    check = None

    # 掃描每一行像素
    for i in range(img.shape[0]-1, 0, -1):
 
        try:
            ean13, check = decode_line(img[i])
        except Exception as e:
            # print(e)
            # print(f"failed on {i}")
            pass
        if check:
            break

    return ean13, check, img


# 解碼單行像素
def decode_line(line):
    bars = read_bars(line)
    left_guard, left_patterns, center_guard, right_patterns, right_guard = classify_bars(
        bars)
    if len(left_patterns) == 0:
        return False, False
    if len(right_patterns) == 0:
        return False, False

    convert_patterns_to_length(left_patterns)
    convert_patterns_to_length(right_patterns)
    left_codes = read_patterns(left_patterns, is_left=True)
    right_codes = read_patterns(right_patterns, is_left=False)
    ean13 = get_ean13(left_codes, right_codes)
    # print("Detected code: " + ean13)
    is_valid = verify(ean13)
    return ean13, is_valid


# 將圖片中的模式轉換為長度
def convert_patterns_to_length(patterns):
    for i in range(len(patterns)):
        patterns[i] = len(patterns[i])


# 讀取模式函式
def read_patterns(patterns, is_left=True):
    # print(len(patterns))
    codes = []
    for i in range(6):
        start_index = i*4
        sliced = patterns[start_index:start_index+4]
        # print(sliced)
        m1 = sliced[0]
        m2 = sliced[1]
        m3 = sliced[2]
        m4 = sliced[3]
        total = m1+m2+m3+m4
        tmp1 = (m1+m2)*1.0
        tmp2 = (m2+m3)*1.0
        at1 = get_AT(tmp1/total)
        at2 = get_AT(tmp2/total)
        if is_left:
            decoded = decode_left(at1, at2, m1, m2, m3, m4)
        else:
            decoded = decode_right(at1, at2, m1, m2, m3, m4)
        codes.append(decoded)
    return codes


# 根據值獲取模式AT
def get_AT(value):
    if value < 2.5/7:
        return 2
    elif value < 3.5/7:
        return 3
    elif value < 4.5/7:
        return 4
    else:
        return 5


# 左半邊解碼函式
def decode_left(at1, at2, m1, m2, m3, m4):
    patterns = {}
    patterns["2,2"] = {"code": "6", "parity": "O"}
    patterns["2,3"] = {"code": "0", "parity": "E"}
    patterns["2,4"] = {"code": "4", "parity": "O"}
    patterns["2,5"] = {"code": "3", "parity": "E"}
    patterns["3,2"] = {"code": "9", "parity": "E"}
    patterns["3,3"] = {"code": "8", "parity": "O", "alter_code": "2"}
    patterns["3,4"] = {"code": "7", "parity": "E", "alter_code": "1"}
    patterns["3,5"] = {"code": "5", "parity": "O"}
    patterns["4,2"] = {"code": "9", "parity": "O"}
    patterns["4,3"] = {"code": "8", "parity": "E", "alter_code": "2"}
    patterns["4,4"] = {"code": "7", "parity": "O", "alter_code": "1"}
    patterns["4,5"] = {"code": "5", "parity": "E"}
    patterns["5,2"] = {"code": "6", "parity": "E"}
    patterns["5,3"] = {"code": "0", "parity": "O"}
    patterns["5,4"] = {"code": "4", "parity": "E"}
    patterns["5,5"] = {"code": "3", "parity": "O"}
    pattern_dict = patterns[str(at1) + "," + str(at2)]
    code = 0
    use_alternative = False
    # 根據條件使用替代模式
    if int(at1) == 3 and int(at2) == 3:
        if m3+1 >= m4:
            use_alternative = True
    if int(at1) == 3 and int(at2) == 4:
        if m2+1 >= m3:
            use_alternative = True
    if int(at1) == 4 and int(at2) == 3:
        if m2+1 >= m1:
            use_alternative = True
    if int(at1) == 4 and int(at2) == 4:
        if m1+1 >= m2:
            use_alternative = True
    if use_alternative:
        code = pattern_dict["alter_code"]
    else:
        code = pattern_dict["code"]
    final = {"code": code, "parity": pattern_dict["parity"]}
    return final


# 右半邊解碼函式
def decode_right(at1, at2, m1, m2, m3, m4):
    patterns = {}
    patterns["2,2"] = {"code": "6"}
    patterns["2,4"] = {"code": "4"}
    patterns["3,3"] = {"code": "8", "alter_code": "2"}
    patterns["3,5"] = {"code": "5"}
    patterns["4,2"] = {"code": "9"}
    patterns["4,4"] = {"code": "7", "alter_code": "1"}
    patterns["5,3"] = {"code": "0"}
    patterns["5,5"] = {"code": "3"}
    pattern_dict = patterns[str(at1) + "," + str(at2)]
    code = 0
    use_alternative = False

    # 根據條件使用替代模式
    if int(at1) == 3 and int(at2) == 3:
        if m3+1 >= m4:
            use_alternative = True
    if int(at1) == 4 and int(at2) == 4:
        if m1+1 >= m2:
            use_alternative = True
    if use_alternative:
        code = pattern_dict["alter_code"]
    else:
        code = pattern_dict["code"]
    final = {"code": code}
    return final


# 讀取條碼像素函式
def read_bars(line):
    bars = []
    current_length = 1
    for i in range(len(line)-1):
        if line[i] == line[i+1]:
            current_length = current_length + 1
        else:
            bars.append(current_length * str(line[i]))
            current_length = 1
    # 移除安全區
    bars.pop(0)
    # print(len(bars))
    return bars


def classify_bars(bars):
    left_guard = bars[0:3]  # 左側檢查碼
    left_patterns = bars[3:27]  # 左側編碼
    center_guard = bars[27:32]  # 中間檢查碼
    right_patterns = bars[32:56]  # 右側編碼
    right_guard = bars[56:59]  # 右側檢查碼
    return left_guard, left_patterns, center_guard, right_patterns, right_guard


def verify(ean13):
    weight = [1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3]
    weighted_sum = 0
    for i in range(12):
        weighted_sum = weighted_sum + weight[i] * int(ean13[i])
    weighted_sum = str(weighted_sum)
    checksum = 0
    units_digit = int(weighted_sum[-1])
    if units_digit != 0:
        checksum = 10 - units_digit
    else:
        checksum = 0
    # print("The checksum of is " + str(checksum))
    if checksum == int(ean13[-1]):
        # print("The code is correct.")
        return True
    else:
        # print("The code is incorrect.")
        return False


def get_ean13(left_codes, right_codes):
    ean13 = ""
    ean13 = ean13 + str(get_first_digit(left_codes))
    for code in left_codes:
        ean13 = ean13 + str(code["code"])
    for code in right_codes:
        ean13 = ean13 + str(code["code"])
    return ean13


def get_first_digit(left_codes):
    parity_dict = {}
    parity_dict["OOOOOO"] = 0
    parity_dict["OOEOEE"] = 1
    parity_dict["OOEEOE"] = 2
    parity_dict["OOEEEO"] = 3
    parity_dict["OEOOEE"] = 4
    parity_dict["OEEOOE"] = 5
    parity_dict["OEEEOO"] = 6
    parity_dict["OEOEOE"] = 7
    parity_dict["OEOEEO"] = 8
    parity_dict["OEEOEO"] = 9
    parity = ""
    for code in left_codes:
        parity = parity + code["parity"]
    return parity_dict[parity]
