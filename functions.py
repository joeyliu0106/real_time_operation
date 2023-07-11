import numpy as np
import scipy.signal as signal
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numba import jit


#===================================
# data preprocessing main part
#===================================
@jit(nopython=False, cache=True)
def AugCoordGen(data, label):
    num, length, width = data.shape

    ###########   random coordinate recording   ###########
    aug_loc = np.zeros((num, 2), dtype=int)

    ###########   random coordinate generating   ###########
    for n in range(num):
        x_rand, y_rand = np.random.randint(low=0, high=[[19], [27]], size=(2, 1), dtype=int)

        label_loc_num = round((label[n, 2] + 1) * (label[n, 3] + 1))
        label_loc = np.zeros((label_loc_num), dtype=int)
        random_loc = np.zeros((36), dtype=int)

        # label location calc
        counter = 0

        for l in range(round(label[n, 3] + 1)):
            for w in range(round(label[n, 2] + 1)):
                label_loc[counter] = round(((label[n, 1] + l) * 24 + (label[n, 0] + w)))
                counter += 1

        # random location calc
        counter = 0

        for l in range(6):
            for w in range(6):
                random_loc[counter] = (y_rand + l) * 24 + (x_rand + w)
                counter += 1

        # overlap checking
        err_count = 0

        for rand in range(36):
            for lab in range(label_loc_num):
                if random_loc[rand] == label_loc[lab]:
                    err_count += 1

        if err_count != 0:
            n = n - 1
        elif err_count == 0:
            aug_loc[n, 0] = x_rand
            aug_loc[n, 1] = y_rand

    return aug_loc


@jit(nopython=False, cache=True)
def OutlierAug(data, label):
    num, length, width = data.shape

    aug_loc = AugCoordGen(data, label)
    aug_data = data.copy()

    for n in range(num):
        x_rand, y_rand = aug_loc[n]

        for l in range(6):
            for w in range(6):
                aug_data[n, y_rand + l, x_rand + w] = 55

    return aug_data


@jit(nopython=False, cache=True)
def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, dtype=np.uint8)

    # return cv2.LUT(src, table), table
    return table

# @jit(nopython=False, cache=True)
def FSN(array, h_bound, l_bound):
    l, w = array.shape
    Fixed = np.zeros(((l, w)), dtype=np.uint8)

    for i in range(l):
        for j in range(w):
            Fixed[i, j] = int((array[i, j] - l_bound) / (h_bound - l_bound) * 255)

            if Fixed[i, j] <= 0:
                Fixed[i, j] = 0
            elif Fixed[i, j] >= 255:
                Fixed[i, j] = 255

    return Fixed


# @jit(nopython=False, cache=True)
def err_map(array, filter):
    l, w = array.shape

    result = signal.convolve2d(array, filter, boundary='symm', mode='same')

    for len in range(l):
        for wid in range(w):
            if abs(result[len, wid]) >= 255:
                result[len, wid] = 255

    return abs(result)


# @jit(nopython=False, cache=True)
# def err_map(data):
#     l, w = data.shape
#     data_pad = np.pad(data, pad_width=(1, 1), mode='edge')
#
#     for i in range(l):
#         for j in range(w):
#             for k in range(3):
#                 for c in range(3):
#                     err_map[i, j] += abs(data_pad[i + 1, j + 1] - data_pad[i + k, j + c])
#
#             err_map[i, j] = err_map[i, j] / 8
#
#     return err_map


def ImageSaving(data):
    image = Image.fromarray(data.astype(np.uint8), 'RGB')
    image.save('temp.jpg')


#===================================
# data processing
#===================================
@jit(nopython=False, cache=True)
def Data_Read(filename):
    data_temp = np.empty((0, 0), float)

    for line in open(f"{filename}.txt"):
        if len(line) > 300:
            data_temp = np.append(data_temp, np.fromstring(line, dtype=float, sep=' '))

    data_temp = data_temp.reshape(-1, 24, 32)
    data_temp = np.rot90(data_temp, k=-1, axes=(1, 2))

    return data_temp


#===================================
# result post-processing
#===================================
@jit(nopython=False, cache=True)
def FileNameExtract(file_name, length):
    file_name.sort(key=lambda x: int(x[:-4]))

    for i in range(length):
        basename = file_name[i]
        file_name[i] = os.path.splitext(basename)[0]

    file_name = np.array(file_name, dtype=int)

    return file_name


@jit(nopython=False, cache=True)
def LabelExtract(file_name, data_len, length):
    labels = np.zeros((data_len, 5), float)

    for i in range(length):
        temp = np.loadtxt("{}.txt".format(file_name[i]), dtype=float, delimiter=' ')

        for j in range(5):
            labels[file_name[i] - 1][j] = temp[j]

    return labels


def LabelTransfer(labels):
    yolo_labels = np.zeros(4, int)

    # yolo_labels[2] = round(labels[3] * 24)
    # yolo_labels[3] = round(labels[4] * 32)
    # yolo_labels[0] = round((labels[1] * 24) - (yolo_labels[2] / 2))
    # yolo_labels[1] = round((labels[2] * 32) - (yolo_labels[3] / 2))

    # testing
    yolo_labels[0] = round(labels[0] * 24)                  # xmin
    yolo_labels[1] = round(labels[1] * 32)                  # ymin
    yolo_labels[2] = round(labels[2] * 24 - yolo_labels[0]) # width
    yolo_labels[3] = round(labels[3] * 32 - yolo_labels[1]) # length

    return yolo_labels


@jit(nopython=False, cache=True)
def LabelCorrect(label, median):
    max_pos = np.zeros((2), dtype=np.uint8)
    label_new = np.zeros((4), dtype=np.uint8)

    value = np.zeros((label[3] + 1, label[2] + 1), dtype=np.float32)

    # kernel label recording
    for l in range(label[3] + 1):
        for w in range(label[2] + 1):
            if label[1] + l > 31:
                break
            elif label[0] + w > 23:
                break

            value[l, w] = median[label[1] + l, label[0] + w]

    # second try
    m = np.argmax(value)
    max_pos[1], max_pos[0] = divmod(m, value.shape[0])

    label_new[0] = label[0]
    label_new[1] = label[1]

    counter = 0

    while (max_pos[0] == 0 or max_pos[0] == label[2] or max_pos[1] == 0 or max_pos[1] == label[3]):
        if max_pos[0] == 0:  # left
            label_new[0] = label_new[0] - 1
            if label_new[0] < 0:
                label_new[0] = 0
                break

        elif max_pos[0] == label[2]:  # right
            label_new[0] = label_new[0] + 1
            if label_new[0] > 23:
                label_new[0] = 23
                break

        elif max_pos[1] == 0:  # up
            label_new[1] = label_new[1] - 1
            if label_new[1] < 0:
                label_new[1] = 0
                break

        elif max_pos[1] == label[3]:  # down
            label_new[1] = label_new[1] + 1
            if label_new[1] > 31:
                label_new[1] = 31
                break

        for l in range(label[3] + 1):
            for w in range(label[2] + 1):
                if label_new[1] + l > 31:
                    break
                elif label_new[0] + w > 23:
                    break

                value[l, w] = median[label_new[1] + l, label_new[0] + w]

        m = np.argmax(value)
        max_pos[1], max_pos[0] = divmod(m, value.shape[0])

        if counter > 20:
            break

        counter += 1

    label_new[2] = label[2]
    label_new[3] = label[3]

    return label_new


@jit(nopython=False, cache=True)
def IOU_calc(data, testing_labels, predicting_labels):
    data_len = len(data)

    intersection = [0] * data_len

    for i in range(data_len):
        counter = 0
        num_t = 0
        num_p = 0
        testing_num = 0
        predicting_num = 0
        testing_iou = np.zeros((100), dtype=int)
        predicting_iou = np.zeros((100), dtype=int)
        for j in range(int(testing_labels[i][3] + 1)):
            for k in range(int(testing_labels[i][2] + 1)):
                testing_iou[num_t] = ((testing_labels[i][1]) + j) * 24 + ((testing_labels[i][0]) + k)
                testing_num += 1
                num_t += 1

        for j in range(predicting_labels[i][3] + 1):
            for k in range(predicting_labels[i][2] + 1):
                predicting_iou[num_p] = (predicting_labels[i][1] + j) * 24 + (predicting_labels[i][0] + k)
                predicting_num += 1
                num_p += 1

        for l in range(testing_num):
            for m in range(predicting_num):

                if testing_iou[l] == predicting_iou[m]:
                    counter += 1
                else:
                    continue

        intersection[i] = counter

    intersection = np.array(intersection, dtype=int)

    area_t = np.zeros((data_len), int)
    area_p = np.zeros((data_len), int)
    iou = np.zeros((data_len + 3), float)
    counter = 0
    iou_sum = 0

    # iou threshold value
    iou_threshold = 0.3

    for i in range(data_len):
        area_t[i] = (testing_labels[i][2] + 1) * (testing_labels[i][3] + 1)
        area_p[i] = (predicting_labels[i][2] + 1) * (predicting_labels[i][3] + 1)
        iou[i] = intersection[i] / (area_t[i] + area_p[i] - intersection[i])
        if iou[i] >= iou_threshold:
            counter += 1
        iou_sum += iou[i]

    iou[data_len] = counter

    # good iou percentage
    # iou[data_len + 1] = counter / data_len
    good_percentage = counter / data_len

    # average iou value
    # iou[data_len + 2] = iou_sum / data_len
    iou_average = iou_sum / data_len

    return (good_percentage, iou_average)


@jit(nopython=False, cache=True)
def HCR(data, label):
    num, length, width = data.shape

    for n in range(num):
        for l in range(length):
            for w in range(width):
                if data[n, l, w] >= 30.5:
                    data[n, l, w] = 0

    result = np.zeros((num), dtype=int)

    for n in range(num):
        value = np.zeros((label[n, 3] + 1, label[n, 2] + 1), dtype=float)

        for l in range(label[n, 3] + 1):
            for w in range(label[n, 2] + 1):
                if label[n, 1] + l > 31:
                    break
                elif label[n, 0] + w > 23:
                    break

                value[l, w] = data[n, label[n, 1] + l, label[n, 0] + w]

        if np.max(value) == np.max(data[n]):
            result[n] = 1
        else:
            continue

    print(result)

    ##############   ratio calc   ################
    counter = 0

    for n in range(num):
        if result[n] == 1:
            counter += 1

    ratio = (counter / num) * 100

    return ratio


def Show_Result(data, labels):
    fig, ax = plt.subplots()

    ax.imshow(data, vmin=22, vmax=32.5, cmap='jet')  # haven't done yet
    rect1 = patches.Rectangle((labels[0], labels[1]), labels[2],labels[3], lw=2, ec='y', fc='none')
    ax.add_patch(rect1)

    plt.show()


def isEmpty(path):
    if os.path.exists(path) and not os.path.isfile(path):
        # Checking if the directory is empty or not
        if not os.listdir(path):
            return True
        else:
            return False
    else:
        print("The path is either for a file or not valid")