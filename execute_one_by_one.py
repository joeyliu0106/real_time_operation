import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.ndimage
import scipy.signal as signal
import cv2
import subprocess
from PIL import Image
from functions import *

np.set_printoptions(linewidth=1000)

if __name__ == 'main':
    # ======================
    # settings
    # ======================
    d_filename = 'data_testing'  # filename of data
    l_filename = 'label_testing'  # filename of labels
    filename = 'training'  # change the name as you want
    gam = 0.33  # gamma for gamma correction
    h_bound = 37.5  # upper bound for FSN
    l_bound = 19.5  # lower bound for FSN
    yolo_d_path = 'C:/Users/Joe/Desktop/streamteck project/yolov5/data/thermal_images'

    ######   data reading   ######
    data = np.empty((0, 0), float)

    for line in open("ThermalData_1.txt"):
        if len(line) > 300:
            data = np.append(data, np.fromstring(line, dtype=float, sep=' '))

    data = data.reshape(-1, 24, 32)
    data = np.rot90(data, k=-1, axes=(1, 2))
    print(data.shape)

    num, l, w = data.shape

    ######   parameters   ######
    # median = np.zeros((num, l, w), dtype=float)
    # norm = np.zeros((num, l, w), dtype=float)

    ######   data preprocessing   ######
    for n in range(num):
        # median filter
        median = scipy.ndimage.median_filter(data[n], size=(3, 3))
        # FSN
        norm = FSN(median, h_bound, l_bound)
        # gamma correction
        gamma, table = gammaCorrection(norm.astype(np.uint8), gam)
        # error map
        filter = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])

        first_err = err_map(gamma)
        second_err = err_map(first_err)
        # channels combining
        err_map_total = np.zeros((l, w, 3), dtype=float)

        err_map_total[:, :, 0] = gamma[:, :].astype(np.uint8)
        err_map_total[:, :, 1] = first_err[:, :].astype(np.uint8)
        err_map_total[:, :, 2] = second_err[:, :].astype(np.uint8)

        path = yolo_d_path
        if not os.path.isdir(path):
            os.mkdir(path)

        os.chdir(path)

        image = Image.fromarray(err_map_total.astype(np.uint8), 'RGB')
        image.save('temp.jpg')


        # ======================
        # yolo prediction
        # ======================
        os.chdir('../yolov5')
        subprocess.run('')



        # ======================
        # result processing
        # ======================
        os.chdir('../yolov5/runs/detect/exp/labels')
        # labels reading
        labels = np.loadtxt('1.txt')
        # labels transfer
        yolo_labels = LabelTransfer(labels=labels)
        # label correction
        corrected_labels = LabelCorrect(yolo_labels, median)
        # result showing
        Show_Result(data[n], corrected_labels)
        
