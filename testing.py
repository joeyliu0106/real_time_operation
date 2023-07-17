from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as patches
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from model import *



# fig = plt.gcf()
# fig.show()
# fig.canvas.draw()

# plt.ion()
# fig, ax = plt.subplots()
#
# n = 0
#
# while True:
#     # compute something
#     plt.plot([1, 2, 3, 4, 5], [2 + n, 1, 2 + n, 1, 2 + n])  # plot something
#
#     # update canvas immediately
#     plt.xlim([0, 10])
#     plt.ylim([0, 10 + n])
#     plt.pause(0.1)
#     fig.canvas.draw()
#     # fig.canvas.flush_events()
#
#     n += 1

model = torch.load('C:/Users/nina/Desktop/streamteck project/real time operation/yolov5/best_new_2.pt')

os.chdir('C:/Users/nina/Desktop/streamteck project/real time operation/yolov5')

im = 'temp.jpg'

results = model(im)

classes, labels = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :4].numpy()
labels = np.squeeze(labels, axis=0)
print('labels: ', labels)
