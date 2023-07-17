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
from numba import jit
import torch
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
d_path = 'C:/Users/nina/Desktop/streamteck project/real time operation/data_testing'     # data path
counter = 1
n = 0
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/nina/Desktop/streamteck project/yolov5/best_new_2.pt', force_reload=True)


######   parameters   ######
if os.path.isdir(detect_path):
    shutil.rmtree(detect_path)


plt.ion()
figure, ax = plt.subplots()

######   data preprocessing   ######
while True:
    print('counter: ', counter)
    start_time = time.time()
    os.chdir(d_path)

    ######   data reading   ######

    data = np.empty((0, 0), float)

    for line in open(f"ThermalData_{n + 1}.txt"):
        if len(line) > 300:
            data = np.append(data, np.fromstring(line, dtype=float, sep=' '))

    data = data.reshape(24, 32)
    data = np.rot90(data, k=-1, axes=(0, 1))

    l, w = data.shape

    # print('dat interval: ', time.time() - start_time)

    # median filter
    # start_time = time.time()

    median = scipy.ndimage.median_filter(data, size=(3, 3))

    # print('med interval: ', time.time() - start_time)
    # FSN
    # start_time = time.time()

    norm = FSN(median, h_bound, l_bound)

    # print('FSN interval: ', time.time() - start_time)

    # gamma correction
    # start_time = time.time()

    # gamma, table = gammaCorrection(norm.astype(np.uint8), gam)
    table = gammaCorrection(norm.astype(np.uint8), gam)
    gamma = cv2.LUT(norm, table)

    # print('gam interval: ', time.time() - start_time)

    # error map
    # start_time = time.time()

    filter = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    first_err = err_map(gamma, filter)
    second_err = err_map(first_err, filter)

    # print('err interval: ', time.time() - start_time)

    # channels combining
    # start_time = time.time()

    err_map_total = np.zeros((l, w, 3), dtype=float)

    err_map_total[:, :, 0] = gamma[:, :].astype(np.uint8)
    err_map_total[:, :, 1] = first_err[:, :].astype(np.uint8)
    err_map_total[:, :, 2] = second_err[:, :].astype(np.uint8)

    # print('com interval: ', time.time() - start_time)

    os.chdir(yolo_path)

    image = Image.fromarray(err_map_total.astype(np.uint8), 'RGB')
    image.save('temp.jpg')


    # ======================
    # yolo prediction
    # ======================
    # start_time = time.time()

    # os.chdir('../yolov5')
    # subprocess.run('python detect.py --weight best_new_2.pt --source temp.jpg --iou-thres 0.0005 --conf-thres 0.0005 --max-det 1 --save-txt --device cpu', shell=True)

    # testing
    image = 'temp.jpg'
    results = model(image)

    classes, labels = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :4].numpy()
    labels = np.squeeze(labels, axis=0)

    # print('pre interval: ', time.time() - start_time)


    # ======================
    # result processing
    # ======================
    # labels transfer
    # start_time = time.time()

    yolo_labels = LabelTransfer(labels=labels)
    print('yolo labels: ', yolo_labels)

    # print('tra interval: ', time.time() - start_time)

    # label correction
    # start_time = time.time()

    corrected_labels = LabelCorrect(yolo_labels, median)
    print('corrected labels: ', corrected_labels)

    # print('cor interval: ', time.time() - start_time)

    # result showing
    # start_time = time.time()

    ax.imshow(data, vmin=22, vmax=32.5, cmap='jet')  # haven't done yet
    rect1 = patches.Rectangle((corrected_labels[0], corrected_labels[1]), corrected_labels[2], corrected_labels[3], lw=2, ec='r', fc='none')
    ax.add_patch(rect1)

    figure.canvas.draw()
    figure.canvas.flush_events()
    rect1.remove()

    print('res interval: ', time.time() - start_time)
    counter += 1
    n += 1