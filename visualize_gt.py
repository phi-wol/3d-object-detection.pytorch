
#%%
%load_ext autoreload
%autoreload 2

#%%
import argparse
import json
import os
from os import path as osp

from tqdm import tqdm
import cv2 as cv
import numpy as np

import torch
import cv2
import copy
from annotation_converters.mmdet.cam_box3d import CameraInstance3DBoxes
from annotation_converters.mmdet.utils import draw_camera_bbox3d_on_img

import mmcv
# plot_rect3d_on_img
# points_cam2img
# draw_camera_bbox3d_on_img

#%% load image
# 'objectron_processed_chair_all/images/chair_batch-36_17_50.jpg'
img = cv2.imread('objectron_processed_chair_all/images/chair_batch-36_17_50.jpg')
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#%% plot boxes
pred_box=torch.tensor([[0.004856228828430176, 0.025137901306152344, 1.720680832862854, 0.5039538145065308, 0.9536432027816772, 0.5375679135322571, 0.8899071216583252, -2.2259745597839355, 2.2209277153015137], [-0.033014535903930664, -0.371906042098999, 2.5649569034576416, 0.5492889881134033, 1.045961856842041, 0.5898192524909973, 0.49425405263900757, -2.897474527359009, 2.8936350345611572], [-0.6709628701210022, -0.4968804121017456, 2.6809558868408203, 0.635883629322052, 0.9161412715911865, 0.6484775543212891, 0.7317734956741333, -2.363032579421997, 2.546478033065796]]) #y
gt_box = torch.tensor([[0.0016107390292674495, 0.02712655255045071, 1.739724733980708, 0.4692857265472412, 0.9392856955528259, 0.5344047546386719, -2.335701110051221, -0.7985773581267579, -0.7583603966889692], [-0.023953137529048263, -0.352381153684326, 2.4048765065534994, 0.5392857193946838, 0.9392856955528259, 0.5344047546386719, -2.5742107705267983, -0.3578900561191358, -0.3368404959688345], [-0.684139442367119, -0.5112269464048129, 2.775640432750393, 0.4692857265472412, 0.9392856955528259, 0.5344047546386719, -2.5943528018004858, -0.2560090029816666, -0.2702874089060428]])

gt_bboxes = CameraInstance3DBoxes(tensor=gt_box, box_dim=gt_box.shape[-1], origin=(0.5,0.5,0.5))
pred_bboxes = CameraInstance3DBoxes(tensor=pred_box, box_dim=pred_box.shape[-1], origin=(0.5,0.5,0.5))
cam_int = np.array([[785.8228759765625, 0.0, 360.1483154296875], [0.0, 785.8228759765625, 470.3028564453125], [0.0, 0.0, 1.0]])

gt_img = draw_camera_bbox3d_on_img(
gt_bboxes, img.copy(), cam_int, None, color=(124,252,0), thickness=5, print_faces=False)

pred_img = draw_camera_bbox3d_on_img(
pred_bboxes, img.copy(), cam_int, None, color=(61, 102, 255), thickness=5, print_faces=False)

output_root = 'viz/'

cv.imwrite(osp.join(output_root, 'gt_viz.jpg'), gt_img)
cv.imwrite(osp.join(output_root, 'pred_viz.jpg'), pred_img)



# %%
