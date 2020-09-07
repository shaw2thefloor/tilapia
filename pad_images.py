import os, shutil
import cv2
from pathlib import Path
import numpy as np


def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def mx_mn_stretch(img):
    mx0 = img[:, :, 0].max()
    mx1 = img[:, :, 1].max()
    mx2 = img[:, :, 2].max()
    mn0 = img[:, :, 0].min()
    mn1 = img[:, :, 1].min()
    mn2 = img[:, :, 2].min()

    # subtract mins
    img[:, :, 0] = img[:, :, 0] - mn0
    img[:, :, 1] = img[:, :, 1] - mn1
    img[:, :, 2] = img[:, :, 2] - mn2

    # normalize
    img[:, :, 0] = img[:, :, 0] / (mx0 - mn0) * 255
    img[:, :, 1] = img[:, :, 1] / (mx1 - mn1) * 255
    img[:, :, 2] = img[:, :, 2] / (mx2 - mn2) * 255

    return img



input_path = Path("C:/Users/fshaw/Documents/PycharmProjects/tilapia/data/raw/small_images")
output_path = Path("C:/Users/fshaw/Documents/PycharmProjects/tilapia/data/raw/padded_images")

for root, dirs, files in os.walk(input_path):
    count = 0
    for file in files:
        if True: #count == 0:
            count += 1
            print(file)
            image_path = str(input_path / file)
            output_image_path = str(output_path / file)

            # print(image_path)

            wb = cv2.SimpleWB()

            imgo = cv2.imread(image_path, 1)
            img = mx_mn_stretch(imgo)
            img = wb.balanceWhite(img)
            #img = white_balance(imgo)
            # get dimension to pad
            shape = img.shape
            if shape[0] > shape[1]:
                pad_dim = 1
            else:
                pad_dim = 0

            # calc pad amount
            minimum = min(shape[0:2])
            target = max(shape[0:2])
            amt = target - minimum
            # print(shape

            pad_color = cv2.mean(img[1:10, 250:260, :])
            # print(pad_color)
            pad_color = tuple([int(x) for x in pad_color])
            # pad image
            img[1:21, 250:260, :] = [0, 0, 255]
            if pad_dim == 1:
                # pad left and right
                out = cv2.copyMakeBorder(img, 0, 0, int(amt / 2), int(amt / 2), cv2.BORDER_CONSTANT, value=pad_color)
            else:
                # pad top and bottom
                out = cv2.copyMakeBorder(img, int(amt / 2), int(amt / 2), 0, 0, cv2.BORDER_CONSTANT, value=pad_color)

            # cv2.imshow("image", out)
            cv2.imwrite(output_image_path, out)
            # waits for user to press any key
            # (this is necessary to avoid Python kernel form crashing)
            # cv2.waitKey(0)

            # closing all open windows
            # cv2.destroyAllWindows()
