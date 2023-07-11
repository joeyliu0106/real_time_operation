from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as patches
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import time
import torch

# os.chdir('C:/Users/nina/Desktop/streamteck project/yolov5')
#
# # Model
# model = torch.hub.load('ultralytics/yolov5', 'best_new_2.pt')  # yolov5n - yolov5x6 official model
# #                                            'custom', 'path/to/best.pt')  # custom model
#
# # Images
# im = 'temp.jpg'  # or file, Path, URL, PIL, OpenCV, numpy, list
#
# # Inference
# results = model(im)
#
# # Results
# results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
# results.xyxy[0]  # im predictions (tensor)
#
# results.pandas().xyxy[0]  # im predictions (pandas)
# #      xmin    ymin    xmax   ymax  confidence  class    name
# # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
#
# results.pandas().xyxy[0].value_counts('name')  # class counts (pandas)
# # person    2
# # tie       1


import torch

os.chdir('C:/Users/nina/Desktop/streamteck project/yolov5')

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/nina/Desktop/streamteck project/yolov5/best_new_2.pt', force_reload=True)   # yolov5n - yolov5x6 official model
#                                            'custom', 'path/to/best.pt')  # custom model

# Images
im = 'temp.jpg'  # or file, Path, URL, PIL, OpenCV, numpy, list

# Inference
# results = model(im)
#
# # Results
# results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
# results.xyxy[0]  # im predictions (tensor)
#
# results.pandas().xyxy[0]  # im predictions (pandas)
# #      xmin    ymin    xmax   ymax  confidence  class    name
# # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
# print(results.pandas().xyxy[0])
# print(type(results))
# #
# # results.pandas().xyxy[0].value_counts('name')  # class counts (pandas)
# # # person    2
# # # tie       1
# #
# # print(results.pandas().xyxy[0].value_counts('name'))
#
#
# # info = results.pandas().xyxy[0].astype(np.float32)
# # print(info)
# # info2 = results.pandas().xyxy[0].to_dict(orient = "records")
# # if len(info2) != 0:
# #     for result in info2:
# #         information = result['xmin', 'ymin', 'xmax', 'ymax']



results = model(im)
labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :4].numpy()

cord_thres = np.squeeze(cord_thres, axis=0)

print(labels)
print(cord_thres.shape)
print(cord_thres)