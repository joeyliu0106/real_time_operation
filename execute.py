import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.ndimage
import scipy.signal as signal
import cv2
import subprocess
from PIL import Image
import shutil
import time
from functions import *

# np.set_printoptions(linewidth=1000)

# ======================
# settings
# ======================
d_filename = 'data_testing'  # filename of data
l_filename = 'label_testing'  # filename of labels
filename = 'testing'  # change the name as you want
gam = 0.33  # gamma for gamma correction
h_bound = 37.5  # upper bound for FSN
l_bound = 19.5  # lower bound for FSN
yolo_path = 'C:/Users/nina/Desktop/streamteck project/yolov5'
detect_path = 'C:/Users/nina/Desktop/streamteck project/yolov5/runs/detect'
counter = 1


######   data reading   ######
# data = np.empty((0, 0), float)
#
# for line in open("ThermalData_1.txt"):
#     if len(line) > 300:
#         data = np.append(data, np.fromstring(line, dtype=float, sep=' '))
#
# data = data.reshape(-1, 24, 32)
# data = np.rot90(data, k=-1, axes=(1, 2))
# print(data.shape)

# test
data = np.load('data_testing.npy') # 957
num, l, w = data.shape
print(data.shape)

######   parameters   ######
# median = np.zeros((num, l, w), dtype=float)
# norm = np.zeros((num, l, w), dtype=float)


plt.ion()
figure, ax = plt.subplots()

######   data preprocessing   ######
for n in range(num):
    start_time = time.time()
    # median filter
    median = scipy.ndimage.median_filter(data[n], size=(3, 3))
    print(median.shape)
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

    os.chdir(yolo_path)

    image = Image.fromarray(err_map_total.astype(np.uint8), 'RGB')
    image.save('temp.jpg')


    # ======================
    # yolo prediction
    # ======================
    os.chdir('../yolov5')
    subprocess.run('python detect.py --weight best_new_2.pt --source temp.jpg --iou-thres 0.0005 --conf-thres 0.0005 --max-det 1 --save-txt --device cpu', shell=True)


    # ======================
    # result processing
    # ======================
    if counter == 1:
        os.chdir(f'{detect_path}/labels')
    elif counter != 1:
        os.chdir(f'{detect_path}{counter}/labels')
    # labels reading
    labels = np.loadtxt('temp.txt')
    # labels transfer
    yolo_labels = LabelTransfer(labels=labels)
    # label correction
    corrected_labels = LabelCorrect(yolo_labels, median)
    # result showing
    ax.imshow(data[n], vmin=22, vmax=32.5, cmap='jet')  # haven't done yet
    rect1 = patches.Rectangle((corrected_labels[0], corrected_labels[1]), corrected_labels[2], corrected_labels[3], lw=2, ec='r', fc='none')
    ax.add_patch(rect1)

    figure.canvas.draw()
    figure.canvas.flush_events()
    # time.sleep(10)
    rect1.remove()

    print('interval: ', time.time() - start_time)

    counter += 1