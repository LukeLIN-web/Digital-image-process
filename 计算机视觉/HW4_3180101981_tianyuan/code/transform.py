import cv2
import numpy as np
import math
import os


def load_eyes_pos(pos_src):
    fr = open(pos_src)
    eyes_pos = np.zeros((2, 2))
    index = 0
    for line in fr.readlines():
        line = line.strip()
        from_line = line.split(',')
        eyes_pos[index, :] = from_line[0:2]
        index += 1
        if index == 2:
            break
    return eyes_pos


def transform(gray_img, eyes_pos):
    x1 = eyes_pos[0][1]
    x2 = eyes_pos[1][1]
    y1 = eyes_pos[0][0]
    y2 = eyes_pos[1][0]
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    angle = math.atan((y2 - y1) / (x2 - x1)) * 180.0 / math.pi
    trans_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    trans_mat[0][2] = trans_mat[0][2] + 37.0 - center[0]
    trans_mat[1][2] = trans_mat[1][2] + 30.0 - center[1]
    rows, cols = gray_img.shape[:2]
    trans_img = cv2.warpAffine(gray_img, trans_mat, (math.floor(cols * 4 / 5), math.floor(rows * 4 / 5)))
    trans_img = cv2.equalizeHist(trans_img)
    return trans_img


def read_data(img_src, write_src):
    for i in range(10):
        path = img_src + str(i + 1) + ".pgm"
        path1 = img_src + str(i + 1) + ".txt"
        image = cv2.imread(path)
        eyes_pos = load_eyes_pos(path1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = transform(image, eyes_pos)
        # cv2.imshow("1", image)
        # cv2.waitKey(100)
        is_exists = os.path.exists(write_src)
        if not is_exists:
            os.makedirs(write_src)
        cv2.imwrite(write_src + str(i + 1) + ".pgm", image)


path = "./att_faces_with_eyes/"
path1 = "./data/"
for i in range(40):
    read_src = path + "s" + str(i+1) + "/"
    write_src = path1 + "s" + str(i+1) + "/"
    read_data(read_src, write_src)
img = cv2.imread("gray.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
my_pos = np.zeros((2, 2))
my_pos[0][0] = 54
my_pos[0][1] = 30
my_pos[1][0] = 53
my_pos[1][1] = 66
img = transform(img, my_pos)
cv2.imwrite("trans.jpg", img)