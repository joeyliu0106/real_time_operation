import subprocess
import os
import numpy as np
import scipy
from functions import *

np.set_printoptions(linewidth=1000)

# os.chdir('../yolov5/')

if __name__ == 'main':
    d_filename = 'data_testing'  # filename of data
    l_filename = 'label_testing'  # filename of labels
    filename = 'training'  # change the name as you want
    gam = 0.33  # gamma for gamma correction
    h_bound = 37.5  # upper bound for FSN
    l_bound = 19.5  # lower bound for FSN
    yolo_d_path = 'C:/Users/nina/Desktop/streamteck project/yolov5/'

    def runcmd(command):
        ret = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1, env='myenv')
        if ret.returncode == 0:
            print("success:",ret)
        else:
            print("error:",ret)

    data = np.empty((0, 0), float)

    for line in open("ThermalData_1.txt"):
        if len(line) > 300:
            data = np.append(data, np.fromstring(line, dtype=float, sep=' '))

    data = data.reshape(-1, 24, 32)
    data = np.rot90(data, k=-1, axes=(1, 2))

    print(data[0])
    print(data.shape)
    num, l, w = data.shape


    median = scipy.ndimage.median_filter(data[0], size=(3, 3))
    print(median.shape)
    # FSN
    norm = FSN(median, h_bound, l_bound)
    print(norm.shape)
    # gamma correction
    gamma, table = gammaCorrection(norm.astype(np.uint8), gam)
    print(gamma.shape)
    filter = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    first_err = err_map(gamma)
    print(first_err.shape)
    second_err = err_map(first_err)
    print(second_err.shape)
    # channels combining
    err_map_total = np.zeros((l, w, 3), dtype=float)

    err_map_total[:, :, 0] = gamma[:, :].astype(np.uint8)
    err_map_total[:, :, 1] = first_err[:, :].astype(np.uint8)
    err_map_total[:, :, 2] = second_err[:, :].astype(np.uint8)
    print(err_map_total.shape)
    os.chdir(yolo_d_path)

    image = Image.fromarray(err_map_total.astype(np.uint8), 'RGB')
    image.save('temp.jpg')

    os.chdir('../yolov5')
    subprocess.run('python detect.py --weight best_new_2.pt --source temp.jpg --iou-thres 0.0005 --conf-thres 0.0005 --max-det 1 --save-txt', shell=True)


    os.chdir('../yolov5/runs/detect/exp/labels')
    # labels reading
    labels = np.loadtxt('temp.txt')
    print(labels)
    print(labels.shape)
    # labels transfer
    yolo_labels = LabelTransfer(labels=labels)
    print(yolo_labels)
    print(yolo_labels.shape)
    # label correction
    corrected_labels = LabelCorrect(yolo_labels, median)
    # result showing
    Show_Result(data[0], corrected_labels)
    # fig, ax = plt.subplots()