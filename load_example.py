
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

from annotation_converters.objectron_helpers import load_annotation_sequence, grab_frames
from objectron.schema import annotation_data_pb2 as annotation_protocol
#  from annotation_converters.objectron_2_coco import save_2_coco

# dummy lists with reduced file number:
lists_root_path = osp.abspath("/Users/philipp.wolters/code/semantic_perception/objectron/index_reduced")
output_folder = "objectron_processed_chair"
fps_divisor = 10
res_divisor = 2

#%%
videos_info = []
avg_vid_len = 0
cl = "bike"
data_root = "objectron_root"
# line = "bike/batch-10/3"
line = "chair/batch-30/14"  # "chair/batch-17/26"
#line = "chair/batch-20/33"
line = "chair/batch-36/17"
#line = "book/batch-1/6"

ann_path = osp.join(data_root, 'annotation' + osp.sep + line.strip() + '.pbdata')

# load a compleete sequence of frames (video)
ann = load_annotation_sequence(ann_path)

# object class can be incorrect in annoatation
for item in ann:
    item[1] = cl # ensure e.g. motobike == bike
assert len(ann) > 0

avg_vid_len += len(ann)
vid_path = osp.join(data_root, 'videos' + osp.sep + line.strip() + osp.sep + 'video.MOV')
videos_info.append((vid_path, ann)) # -> param for obj2coco
# list of ann_sequnces and vid_paths for all classes

avg_vid_len /= len(videos_info)

#%%
def get_frame_annotation(sequence, frame_id):
    """Grab an annotated frame from the sequence."""
    data = sequence.frame_annotations[frame_id]
    object_id = 0
    object_keypoints_2d = []
    object_keypoints_3d = []
    object_rotations = []
    object_translations = []
    object_scale = []
    num_keypoints_per_object = []
    object_categories = []
    annotation_types = []
    unit_points = []
    # Get the camera for the current frame. We will use the camera to bring
    # the object from the world coordinate to the current camera coordinate.
    camera = np.array(data.camera.transform).reshape(4, 4)
    view_matrix = np.array(data.camera.view_matrix).reshape(4, 4)
    intrinsics = np.array(data.camera.intrinsics).reshape(3, 3)

    for obj in sequence.objects: # what happens to objects not visible in the scene?
        rotation = np.array(obj.rotation).reshape(3, 3)
        translation = np.array(obj.translation)
        # scale invariant
        object_scale.append(np.array(obj.scale))
        transformation = np.identity(4)
        transformation[:3, :3] = rotation
        transformation[:3, 3] = translation
        # transformation into camera coordinates
        obj_cam = np.matmul(view_matrix, transformation)
        object_translations.append(obj_cam[:3, 3])
        object_rotations.append(obj_cam[:3, :3])

        # transfer rotation matrix to euler angles
        object_categories.append(obj.category)
        annotation_types.append(obj.type)
        unit_points.append(obj.keypoints) # contains box vertices in the "BOX" coordinate, (i.e. it's a unit box)

        # we need translation, scale & all three rotations

    keypoint_size_list = []
    for annotations in data.annotations:
        num_keypoints = len(annotations.keypoints)
        keypoint_size_list.append(num_keypoints)
        for keypoint_id in range(num_keypoints):
            keypoint = annotations.keypoints[keypoint_id]
            object_keypoints_2d.append(
                (keypoint.point_2d.x, keypoint.point_2d.y, keypoint.point_2d.depth))
            object_keypoints_3d.append(
                (keypoint.point_3d.x, keypoint.point_3d.y, keypoint.point_3d.z))
        num_keypoints_per_object.append(num_keypoints)
        object_id += 1

    # return [object_keypoints_2d, object_categories, keypoint_size_list,
    #         annotation_types]
    return [object_categories, object_scale, object_rotations, object_translations, intrinsics,
    object_keypoints_2d, keypoint_size_list, annotation_types, object_keypoints_3d, unit_points]

#%% def load_annotation_sequence(annotation_file):
annotation_file = ann_path

frame_annotations = []
with open(annotation_file, 'rb') as pb:
    sequence = annotation_protocol.Sequence()
    sequence.ParseFromString(pb.read()) # read protobuf
    for i in range(len(sequence.frame_annotations)):
        frame_annotations.append(get_frame_annotation(sequence, i)) # TODO: change frame annotation
        # annotation, cat, num_keypoints, types

#%%
sequence.frame_annotations
sequence.objects

# sequence.frame_annotations[1].camera.intrinsics
# sequence.objects[0].keypoints[1]

#%%
def decode_keypoints(keypoints, keypoint_size_list, size):
    # decode keypoints from 3D to 2D
    keypoints = np.split(keypoints, np.array(np.cumsum(keypoint_size_list)))
    keypoints = [points.reshape(-1, 3) for points in keypoints]
    unwrap_mat = np.asarray([size[0], size[1], 1.], np.float32)
    keypoints = [
        np.multiply(keypoint, unwrap_mat).astype(int)[:, :-1]
            for keypoint in keypoints
    ][:len(keypoint_size_list)]
    for i, kp in enumerate(keypoints):
        assert len(kp) == keypoint_size_list[i]
        assert len(kp) == OBJECTRON_NUM_KPS
    return keypoints

# TODO: convert bounding box representation:
def decode_bbox(keypoints, keypoint_size_list, size):
    pass

#TODO: or rewrite this class to 3D versions
def get_bboxes_from_keypoints(keypoints, num_objects, size, clip_bboxes=False):
    w, h = size
    bboxes = []
    num_valid = 0

    for i in range(num_objects):
        min_x = np.min(keypoints[i][:,0])
        min_y = np.min(keypoints[i][:,1])
        max_x = np.max(keypoints[i][:,0])
        max_y = np.max(keypoints[i][:,1])
        if clip_bboxes:
            min_x, min_y = max(0, min_x), max(0, min_y)
            max_x, max_y = min(w - 1, max_x), min(h - 1, max_y)
        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
        if min_x < 0 or min_y < 0 or max_x >= w or max_y >= h or bbox[2]*bbox[3] == 0:
            bboxes.append(None)
        else:
            bboxes.append(bbox)
            num_valid += 1

    if num_valid > 0:
        return bboxes

    return None

def np_encoder(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    return None
#%%
def save_2_coco(output_root, subset_name, data_info, obj_classes, fps_divisor,
                res_divisor, dump_images=False, clip_classes=[], debug=False):
    json_name = f'objectron_{subset_name}.json'
    ann_folder = osp.join(output_root, 'annotations')
    img_folder = osp.join(output_root, 'images')
    if not osp.isdir(ann_folder):
        os.makedirs(ann_folder, exist_ok=True)
    if not osp.isdir(img_folder):
        os.makedirs(img_folder, exist_ok=True)

    img_id = 0
    ann_id = 0

    stat = {'Total frames' : 0, 'Avg box size': [0, 0], 'Total boxes': 0}
    ann_dict = {}
    categories = [{"id": i + 1, "name": cl} for i, cl in enumerate(obj_classes)]
    class_2_id = {cl : i + 1 for i, cl in enumerate(obj_classes)}
    images_info = []
    annotations = []

    for item in tqdm(data_info):
        vid_path, annotation = item
        # assert get_video_frames_number(vid_path) == len(annotation)
        req_frames = []
        for frame_idx in range(len(annotation)):
            if frame_idx % fps_divisor == 0:
                req_frames.append(frame_idx)
        frames = grab_frames(vid_path, req_frames, False)

        for frame_idx, frame_ann in enumerate(annotation):
            if frame_idx not in frames:
                continue
            if frames[frame_idx] is None:
                print('Warning: missing frame in ' + vid_path)
                continue
            #object_keypoints_2d, object_categories, keypoint_size_list, annotation_types

            h, w = frames[frame_idx].shape[0] // res_divisor, frames[frame_idx].shape[1] // res_divisor
            keypoints = decode_keypoints(frame_ann[0], frame_ann[2], (w, h))
            num_objects = len(frame_ann[2])
            bboxes = get_bboxes_from_keypoints(keypoints, num_objects, (w, h),
                                               clip_bboxes=frame_ann[1] in clip_classes)
            if bboxes is None:
                continue

            image_info = {}
            image_info['id'] = img_id
            img_id += 1
            image_info['height'], image_info['width'] = h, w
            vid_name_idx = vid_path.find('batch-')
            image_info['file_name'] = osp.join('images',
                    frame_ann[1] + '_' + vid_path[vid_name_idx : vid_path.rfind(osp.sep)].replace(osp.sep, '_') + \
                    '_' + str(frame_idx) + '.jpg')
            images_info.append(image_info)
            stat['Total frames'] += 1

            if debug:
                # visual debug
                frames[frame_idx] = cv.resize(frames[frame_idx], (w, h))
                for kp_pixel in keypoints[0]:
                    cv.circle(frames[frame_idx], (kp_pixel[0], kp_pixel[1]), 5, (255, 0, 0), -1)
                if len(keypoints) > 1:
                    for kp_pixel in keypoints[1]:
                        cv.circle(frames[frame_idx], (kp_pixel[0], kp_pixel[1]), 5, (0, 0, 255), -1)
                for bbox in bboxes:
                    if bbox is not None:
                        cv.rectangle(frames[frame_idx], (bbox[0], bbox[1]),
                                    (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 1)
                cv.imwrite(osp.join(output_root, image_info['file_name']), frames[frame_idx])

            if dump_images and not debug:
                frames[frame_idx] = cv.resize(frames[frame_idx], (w, h))
                cv.imwrite(osp.join(output_root, image_info['file_name']), frames[frame_idx])

            for i in range(num_objects):
                if bboxes[i] is not None:
                    stat['Total boxes'] += 1
                    stat['Avg box size'][0] += bboxes[i][2]
                    stat['Avg box size'][1] += bboxes[i][3]
                    ann = {
                        'id': ann_id,
                        'image_id': image_info['id'],
                        'segmentation': [],
                        'num_keypoints': frame_ann[2][i],
                        'keypoints': list(keypoints[i].reshape(-1)),
                        'category_id': class_2_id[frame_ann[1]],
                        'iscrowd': 0,
                        'area': bboxes[i][2] * bboxes[i][3],
                        'bbox': bboxes[i]
                        }
                    ann_id += 1
                    annotations.append(ann)

    ann_dict['images'] = images_info
    #annbev_direction (str): Flip direction (horizontal or vertical)._dict['categories'] = categories
    ann_dict['annotations'] = annotations
    with open(osp.join(ann_folder, json_name), 'w') as f:
        f.write(json.dumps(ann_dict, default=np_encoder))
    stat['Avg box size'][0] /= stat['Total boxes']
    stat['Avg box size'][1] /= stat['Total boxes']
    return stat
#%%

ALL_CLASSES = ['bike', 'book', 'bottle', 'cereal_box', 'camera', 'chair', 'cup', 'laptop', 'shoe']
OBJECTRON_NUM_KPS = 9

subset = "train"
obj_classes = ["bike"]
data_info = {}

#videos_info, avg_len = load_video_info(args.data_root, subset, args.obj_classes)
#print(f'# of {subset} videos: {len(videos_info)}, avg length: {avg_len}')
data_info[subset] = videos_info

# #%%
# for k in data_info: #k: subset
#     print('Converting ' + k)
#     stat = save_2_coco(output_folder, k, data_info[k], obj_classes,
#                         fps_divisor, res_divisor, True, ['shoe', 'bike'], True)

# %% Visualize GT

frame_id =  50 # 50 #
# with open(annotation_file, 'rb') as pb:
#     sequence = annotation_protocol.Sequence()
#     sequence.ParseFromString(pb.read())
vid_path, annotation = data_info[subset][0] # für alle videos -> das erste video
frame = grab_frames(vid_path, [frame_id], True)

#%% specific annotations for one frame
from objectron.dataset import graphics
import matplotlib.pyplot as plt

#annotation, cat, num_keypoints, types = get_frame_annotation(sequence, frame_id)
image = graphics.draw_annotation_on_image(frame[frame_id].copy(), annotation[frame_id][0], annotation[frame_id][2])
imgplot = plt.imshow(image)

#%%


# %%

# %%
# show_result_meshlab(
#     data,
#     result,
#     args.out_dir,
#     args.score_thr,
#     show=args.show,
#     snapshot=args.snapshot,
#     task='mono-det')

# file_name = show_proj_det_result_meshlab(data, result, out_dir,
#                                                  score_thr, show, snapshot)


#%%
# if 'cam2img' not in data['img_metas'][0][0]:
#             raise NotImplementedError(
#                 'camera intrinsic matrix is not provided')

#show_bboxes = CameraInstance3DBoxes(pred_bboxes, box_dim=pred_bboxes.shape[-1], origin=(0.5, 1.0, 0.5))

# show_multi_modality_result(
#     img,
#     None,
#     show_bboxes,
#     data['img_metas'][0][0]['cam2img'],
#     out_dir,
#     file_name,
#     box_mode='camera',
#     show=show)

#%%
def draw_camera_bbox3d_on_img(bboxes3d, # CameraInstance3DBoxes`, shape=[M, 9] TODO: change to 9
                              raw_img,
                              cam2img, # intrinsics
                              img_metas,
                              color=(0, 255, 0),
                              thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`CameraInstance3DBoxes`, shape=[M, 7]): 
            3d bbox in camera coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        cam2img (dict): Camera intrinsic matrix,
            denoted as `K` in depth bbox coordinate system.
        img_metas (dict): Useless here.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    #from mmdet3d.core.bbox import points_cam2img

    img = raw_img.copy()
    cam2img = copy.deepcopy(cam2img)
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)
    if not isinstance(cam2img, torch.Tensor):
        cam2img = torch.from_numpy(np.array(cam2img))

    assert (cam2img.shape == torch.Size([3, 3])
            or cam2img.shape == torch.Size([4, 4]))
    cam2img = cam2img.float().cpu()

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(points_3d, cam2img)
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)

def draw_camera_bbox3d_on_img_adjusted(bboxes3d, # CameraInstance3DBoxes`, shape=[M, 9] TODO: change to 9
                                        raw_img,
                                        cam2img, # intrinsics
                                        img_metas,
                                        color=(0, 255, 0),
                                        thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`CameraInstance3DBoxes`, shape=[M, 7]): 
            3d bbox in camera coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        cam2img (dict): Camera intrinsic matrix,
            denoted as `K` in depth bbox coordinate system.
        img_metas (dict): Useless here.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    #from mmdet3d.core.bbox import points_cam2img

    img = raw_img.copy()
    cam2img = copy.deepcopy(cam2img)
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)

    # Version 1: Before projection
    # negate z in points
    points_3d[:, 2] = - points_3d[:, 2] 
    # swap x & y in points
    points_3d[:, [0, 1]] = points_3d[:, [1, 0]]
    # sawp px & py in intrinsics
    cam2img[0,2], cam2img[1,2] = cam2img[1,2], cam2img[0,2]

    print(points_3d)
    print(cam2img)
    if not isinstance(cam2img, torch.Tensor):
        cam2img = torch.from_numpy(np.array(cam2img))

    assert (cam2img.shape == torch.Size([3, 3])
            or cam2img.shape == torch.Size([4, 4]))
    cam2img = cam2img.float().cpu()

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(points_3d, cam2img)
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)

def draw_objectron(corners, # CameraInstance3DBoxes`, shape=[M, 9] TODO: change to 9
                              raw_img,
                              cam2img, # intrinsics
                              img_metas,
                              color=(0, 255, 0),
                              thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`CameraInstance3DBoxes`, shape=[M, 7]): 
            3d bbox in camera coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        cam2img (dict): Camera intrinsic matrix,
            denoted as `K` in depth bbox coordinate system.
        img_metas (dict): Useless here.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    #from mmdet3d.core.bbox import points_cam2img

    img = raw_img.copy()
    cam2img = copy.deepcopy(cam2img)
    corners_3d = corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)
    if not isinstance(cam2img, torch.Tensor):
        cam2img = torch.from_numpy(np.array(cam2img))

    assert (cam2img.shape == torch.Size([3, 3])
            or cam2img.shape == torch.Size([4, 4]))
    cam2img = cam2img.float().cpu()

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(points_3d, cam2img)
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)

#%% points_cam2img
#from mmdet3d.core.utils import array_converter

#@array_converter(apply_to=('points_3d', 'proj_mat'))
def points_cam2img(points_3d, proj_mat, with_depth=False):
    """Project points in camera coordinates to image coordinates.

    Args:
        points_3d (torch.Tensor | np.ndarray): Points in shape (N, 3)
        proj_mat (torch.Tensor | np.ndarray):
            Transformation matrix between coordinates.
        with_depth (bool, optional): Whether to keep depth in the output.
            Defaults to False.

    Returns:
        (torch.Tensor | np.ndarray): Points in image coordinates,
            with shape [N, 2] if `with_depth=False`, else [N, 3].
    """
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1

    assert len(proj_mat.shape) == 2, 'The dimension of the projection'\
        f' matrix should be 2 instead of {len(proj_mat.shape)}.'
    d1, d2 = proj_mat.shape[:2]
    assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (
        d1 == 4 and d2 == 4), 'The shape of the projection matrix'\
        f' ({d1}*{d2}) is not supported.'
    if d1 == 3:
        proj_mat_expanded = torch.eye(
            4, device=proj_mat.device, dtype=proj_mat.dtype)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    # previous implementation use new_zeros, new_one yields better results
    points_4 = torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)

    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        point_2d_res = torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)

    return point_2d_res

#%% plot_rect3d_on_img
def plot_rect3d_on_img(img,
                       num_rects,
                       rect_corners,
                       color=(0, 255, 0),
                       thickness=1):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    top_indices = ((0,5), (1, 4), (0, 7), (3, 4))
    color_2 = (50, 170, 255)
    for i in range(num_rects):
        corners = rect_corners[i].astype(int)
        for start, end in line_indices:
            cv2.line(img, (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]), color, thickness,
                     cv2.LINE_AA)
        for start, end in top_indices:
            cv2.line(img, (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]), color_2, thickness,
                     cv2.LINE_AA)

    return img.astype(np.uint8)

#%%

#%%
test = """Convert multi-modality detection results into 2D results.

    Project the predicted 3D bbox to 2D image plane and visualize them.

    Args:
        img (np.ndarray): The numpy array of image in cv2 fashion.
        gt_bboxes (:obj:`BaseInstance3DBoxes`): Ground truth boxes.
        pred_bboxes (:obj:`BaseInstance3DBoxes`): Predicted boxes.
        proj_mat (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        out_dir (str): Path of output directory.
        filename (str): Filename of the current frame.
        box_mode (str, optional): Coordinate system the boxes are in.
            Should be one of 'depth', 'lidar' and 'camera'.
            Defaults to 'lidar'.
        img_metas (dict, optional): Used in projecting depth bbox.
            Defaults to None.
        show (bool, optional): Visualize the results online. Defaults to False.
        gt_bbox_color (str or tuple(int), optional): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (255, 102, 61).
        pred_bbox_color (str or tuple(int), optional): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (72, 101, 241).
    """
#%% frame_anntations: for all frames: 310
# [object_categories, 
# object_scale,
# object_rotations, 
# object_translations, 
# intrinsics,
# object_keypoints_2d, 
# keypoint_size_list, 
# annotation_types,
# object_keypoints_3d
# unit box]
index = frame_id
object_scales = frame_annotations[index][1]
object_rotations =  frame_annotations[index][2]

object_translations =  frame_annotations[index][3]
# object translation: offset to centroid of object?
intrinsics = frame_annotations[index][4]
object_keypoints_2d = frame_annotations[index][5]
object_keypoints_3d = frame_annotations[index][8]
unit_points = frame_annotations[index][9]
upoints3d = np.array([[kp.x, kp.y, kp.z] for kp in unit_points[0]])

gt_bbox_color=(61, 102, 255)
pred_bbox_color=(241, 101, 72)
img_metas = None

#%%
from annotation_converters.objectron_box import Box
from annotation_converters.graphics import draw_annotation_on_image

test_box = Box.from_transformation(object_rotations[0], object_translations[0], object_scales[0])
test_box.vertices

test_box_zr = Box.from_transformation(np.identity(3), object_translations[0], object_scales[0])

# draw with original keypoints_2d, already projected in the image plane
image = graphics.draw_annotation_on_image(frame[frame_id].copy(), annotation[frame_id][0], annotation[frame_id][2])
imgplot = plt.imshow(image)

#%%
from annotation_converters.mmdet.cam_box3d import CameraInstance3DBoxes

from scipy.spatial.transform import Rotation as R
r = R.from_matrix(object_rotations[0])
# what rotation representation do I need?
object_euler_angles = r.as_euler('xyz', degrees=False)
print(r.as_euler('xyz', degrees=True))

gt_bboxes = np.concatenate([object_translations[0], object_scales[0], object_euler_angles])
gt_bboxes = torch.tensor([gt_bboxes])
gt_bboxes = CameraInstance3DBoxes(gt_bboxes, box_dim=gt_bboxes.shape[-1], origin=(0.5,0.5,0.5))
gt_bboxes.corners

#%%
test_box_obj = Box.from_transformation(object_rotations[0], object_translations[0], object_scales[0])
test_box_obj.vertices[[1,2, 4, 3, 5, 6, 8 ,7]] 


#%% create vertices from unit box
def create_Box(translation, rotation, scale):
    box_transformation = np.identity(4)
    box_transformation[:3, :3] = rotation
    box_transformation[:3, 3] = translation

    points_scaled = upoints3d * scale.T

    points_hom = np.concatenate((points_scaled, np.ones_like(points_scaled[:, :1])), axis=-1).T
    box_vertices_3d_world = np.matmul(box_transformation, points_hom) 
    return box_vertices_3d_world #[:3].T

#%%
create_Box(object_translations[0], object_rotations[0], object_scales[0])
#%%
rotation = np.array(sequence.objects[0].rotation).reshape(3, 3)
translation = np.array(sequence.objects[0].translation)
scale = np.array(sequence.objects[0].scale)
p_3d = create_Box(translation, rotation, scale)


# np.array(sequence.frame_annotations[index].camera.intrinsics).reshape(3,3)
view = np.array(sequence.frame_annotations[index].camera.view_matrix).reshape(4,4)
proj = np.array(sequence.frame_annotations[index].camera.projection_matrix).reshape(4,4)

np.matmul(view, p_3d)[:3].T

#%% keypoints = data[features.FEATURE_NAMES['POINT_2D']]
image = draw_annotation_on_image(frame[index].copy(), object_keypoints_2d, [9]) # test_box_zr.vertices
imgplot = plt.imshow(image)

#%%
corner_points_3d_mapped = points_cam2img(np.array(object_keypoints_3d), np.array(intrinsics))
image = draw_annotation_on_image(frame[index].copy(), corner_points_3d_mapped, [9]) 
imgplot = plt.imshow(image)

#%%
draw_bbox = draw_objectron
gt_bboxes = torch.tensor([test_box_zr.vertices[1:]]).float()


#%%
# points_3d are transformed Box coordinates in Camera Coordinates
# project 3d points to image plane

def project_by_intrinsics(vertices_3d, intr):
    """
    Project using camera intrinsics.

    Objectron frame (x down, y right, z in); 
    H-Z frame (x right, y down, z out); 
    Objectron intrinsics has px and py swapped;
    px and py are from original image size (1440, 1920);

    Approach 1:
    To transform Objectron frame to H-Z frame,
    we need to negate z and swap x and y;
    To modify intrinsics, we need to swap px, py.

    Or alternatively, approach 2:
    we change the sign for z and swap x and y after projection.
    
    Reference
    https://github.com/google-research-datasets/Objectron/issues/39#issuecomment-835509430
    https://amytabb.com/ts/2019_06_28/
    """
    # Objectron to H-Z frame
    vertices_3d[:, 2] = - vertices_3d[:, 2]    
    # scale intrinsics from (1920, 1440) to (640, 480)
    #intr[:2, :] = intr[:2, :] / np.array([[1920],[1440]]) * np.array([[640],[480]])
    point_2d = intr @ vertices_3d.T 
    point_2d[:2, :] = point_2d[:2, :] / point_2d[2, :]
    # landscape to portrait swap x and y.
    point_2d[[0, 1], :] = point_2d[[1, 0], :]
    arranged_points = point_2d.T[:, :2]
    return arranged_points

def draw_box_intrinsics(img, arranged_points):
    """
    plot arranged_points on img.
    arranged_points: list of points [[x, y]] in image coordinate. 
    """
    RADIUS = 10
    COLOR = (61, 102, 255) #(255, 255, 255)
    EDGES = [
      [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
      [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
      [1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
    ] 
    for i in range(arranged_points.shape[0]):
        x, y = arranged_points[i]
        cv2.circle(img, (int(x), int(y)), RADIUS, COLOR, -10)
    for edge in EDGES:
        start_points = arranged_points[edge[0]]
        start_x = int(start_points[0])
        start_y = int(start_points[1])
        end_points = arranged_points[edge[1]]
        end_x = int(end_points[0])
        end_y = int(end_points[1])
        cv2.line(img, (start_x, start_y), (end_x, end_y), COLOR, 10)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def project_by_intrinsics_alt(vertices_3d, intr):
    """
    Project using camera intrinsics.

    Objectron frame (x down, y right, z in); 
    H-Z frame (x right, y down, z out); 
    Objectron intrinsics has px and py swapped;
    px and py are from original image size (1440, 1920);

    Approach 1:
    To transform Objectron frame to H-Z frame,
    we need to negate z and swap x and y;
    To modify intrinsics, we need to swap px, py.

    Or alternatively, approach 2:
    we change the sign for z and swap x and y after projection.
    
    Reference
    https://github.com/google-research-datasets/Objectron/issues/39#issuecomment-835509430
    https://amytabb.com/ts/2019_06_28/
    """
    # Objectron to H-Z frame
    vertices_3d[:, 2] = - vertices_3d[:, 2]
    # scale intrinsics from (1920, 1440) to (640, 480)
    #intr[:2, :] = intr[:2, :] / np.array([[1920],[1440]]) * np.array([[640],[480]])
    point_2d = intr @ vertices_3d.T 
    point_2d[:2, :] = point_2d[:2, :] / point_2d[2, :]
    # landscape to portrait swap x and y.
    point_2d[[0, 1], :] = point_2d[[1, 0], :]
    arranged_points = point_2d.T[:, :2]
    return arranged_points

#%%
pt2d_intr_original = project_by_intrinsics(np.array(object_keypoints_3d).reshape(9, 3), intrinsics_2.copy())
draw_box_intrinsics(frame[index].copy(), pt2d_intr_original)

#%%
centroid = (gt_bboxes.corners.max(axis=1).values + gt_bboxes.corners.min(axis=1).values)/2
dof9 = np.concatenate([centroid, gt_bboxes.corners[0][[0, 1, 3, 2, 4, 5, 7, 6]]])

#%%
pt2d_intr = project_by_intrinsics(test_box.vertices.copy(), intrinsics.copy())
draw_box_intrinsics(frame[index].copy(), pt2d_intr)

#%%
pt2d_intr = project_by_intrinsics(dof9.copy(), intrinsics.copy())
draw_box_intrinsics(frame[index].copy(), pt2d_intr)


#%%
pt2d_intr = project_by_intrinsics_alt(test_box.vertices, intrinsics.copy())

#%%
# (x, y, z, x_size, y_size, z_size, yaw, ...)
gt_bboxes = CameraInstance3DBoxes(gt_bboxes, box_dim=gt_bboxes.shape[-1], origin=(0.5,0.5,0.5))
pred_bboxes = None

#%%

#%% 
import mmcv
from mmdet.cam_box3d import CameraInstance3DBoxes

draw_bbox = draw_camera_bbox3d_on_img
draw_bbox = draw_camera_bbox3d_on_img_adjusted

img = frame[frame_id].copy()

out_dir = "test_output"
result_path = out_dir
proj_mat = intrinsics.copy()

filename = "test_visualization"
box_mode='camera'
show=False

if img is not None:
    mmcv.imwrite(img, osp.join(result_path, f'{filename}_{index}_img.png'))

if gt_bboxes is not None:
    gt_img = draw_bbox(
        gt_bboxes, img, proj_mat, img_metas, color=gt_bbox_color, thickness=5)
    mmcv.imwrite(gt_img, osp.join(result_path, f'{filename}_{index}_gt.png'))

bounding_box_dummy = None
pred_bboxes = None

if pred_bboxes is not None:
    pred_img = draw_bbox(
        pred_bboxes, img, proj_mat, img_metas, color=pred_bbox_color)
    mmcv.imwrite(pred_img, osp.join(result_path, f'{filename}_pred.png'))

# %%

# %%
image = graphics.draw_annotation_on_image(frame[frame_id].copy(), annotation[frame_id][0], annotation[frame_id][2])
imgplot = plt.imshow(image)
















# %%

#%% ###########################################################################################
# compare without rotation
test_box_zr = Box.from_transformation(np.identity(3), object_translations[0], object_scales[0])
test_box_zr.vertices[[1,2, 4, 3, 5, 6, 8 ,7]] 

#%% compare without rotation 

gt_bboxes = np.concatenate([object_translations[0], object_scales[0], np.array([0])])
gt_bboxes = torch.tensor([gt_bboxes])
gt_bboxes = CameraInstance3DBoxes(gt_bboxes, box_dim=gt_bboxes.shape[-1], origin=(0.5,0.5,0.5))
gt_bboxes.corners
# the same :)

#%% with rotation

obj_rot = object_rotations[0]
rot_y = np.array(
    [[0, 0, 1], 
    [0, 1, 0], 
    [-1, 0 , 0]]) # rot y

rot_z = R.from_rotvec(np.array([0, 0, np.pi/2]), degrees=False).as_matrix() # rot_z

rot_y = R.from_rotvec(np.array([0, 1, 0]), degrees=False).as_matrix() # rot_y

rott = R.from_rotvec(np.array([1, 2, 3]), degrees=False).as_matrix()

rott = np.identity(3)

rot_x = R.from_rotvec(np.array([np.pi/2, 0, 0]), degrees=False).as_matrix() # rot_x

rott = np.matmul(rot_z, np.matmul(rot_y, rot_x))

test_box_fr = Box.from_transformation(obj_rot, np.array([1.,2.,3.]), np.array([10.,20.,30.]))
test_box_fr.vertices[[1, 2, 4, 3, 5, 6, 8 ,7]] 

#%% with euler angles

r = R.from_matrix(object_rotations[0])
# what rotation representation do I need?

object_euler_angles = np.array([0, 0, np.pi/2]) #z
object_euler_angles = np.array([0, np.pi/2, 0]) #y

object_euler_angles = zero_rot = np.array([0, 0, 0]) # zero
object_euler_angles = np.array([np.pi/2, 0, 0]) #x

object_euler_angles = np.array([np.pi/2, np.pi/2, np.pi/2])

object_euler_angles = r.as_euler('xyz', degrees=False) * np.array([1, 1, 1])
print(object_euler_angles)

gt_bboxes = np.concatenate([np.array([1.,2.,3.]), np.array([10.,20.,30.]), object_euler_angles])
gt_bboxes = torch.tensor([gt_bboxes])
gt_bboxes = CameraInstance3DBoxes(gt_bboxes, box_dim=gt_bboxes.shape[-1], origin=(0.5,0.5,0.5))
gt_bboxes.corners

#%% keypoints = data[features.FEATURE_NAMES['POINT_2D']]
image = draw_annotation_on_image(frame[index].copy(), object_keypoints_2d, [9]) # test_box_zr.vertices
imgplot = plt.imshow(image)

#%%
image = draw_annotation_on_image(frame[index].copy(), test_box_zr.vertices, [9]) 
imgplot = plt.imshow(image)

#%% check calculation
import math
def rotationMatrixToEulerAngles(R) :
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])
object_euler_angles = rotationMatrixToEulerAngles(object_rotations[0])

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

eulerAnglesToRotationMatrix(object_euler_angles)

#%%
from mmdet.utils import rotation_3d_in_axis

rotation_3d_in_axis(points = torch.tensor([[[ -5., -10., -15.]]]), angles = torch.tensor([np.pi/2]), axis = 0)

#%%
np.dot(R.from_rotvec(np.array([0, 0, np.pi/2]), degrees=False).as_matrix(), np.array([1,1,1]))

#%% Kitti rotation in x:
angles = torch.tensor([np.pi/2])
rot_sin = torch.sin(angles)
rot_cos = torch.cos(angles)
ones = torch.ones_like(rot_cos)
zeros = torch.zeros_like(rot_cos)

rot_mat_T = torch.stack([
                torch.stack([ones, zeros, zeros]),
                torch.stack([zeros, rot_cos, rot_sin]),
                torch.stack([zeros, -rot_sin, rot_cos])
            ])
# vs R.from_rotvec(np.array([np.pi/2, 0, 0]), degrees=False).as_matrix()

#%%
centroid = (gt_bboxes.corners.max(axis=1).values + gt_bboxes.corners.min(axis=1).values)/2
dof9 = np.concatenate([centroid, gt_bboxes.corners[0]])
Box.fit(dof9)

###########################################################################################


#%%
from scipy.spatial.transform import Rotation as R
rotation_original = np.array(sequence.objects[0].rotation).reshape(3,3)
#rotation_original = np.eye(3)
r_before = R.from_matrix(rotation_original)
object_euler_angles = r_before.as_euler('xyz', degrees=False)

print(rotation_original)
print(object_euler_angles )

#%%
R.from_rotvec(object_euler_angles.tolist()).as_matrix()
R.from_euler("xyz", object_euler_angles.tolist()).as_matrix()


#%%
print(object_euler_angles)
r_after = R.from_euler('xyz', object_euler_angles)
print(r_before.as_matrix())
print(r_after.as_matrix())
print(r_before.as_matrix()-r_after.as_matrix())
r_after.as_euler('xyz', degrees=True)

#%%
intrinsics[:2] /= res_divisor

# np.array(sequence.objects[0].rotation).reshape(3,3)
# =/ R.from_matrix(np.array(sequence.objects[0].rotation).reshape(3,3)).as_matrix()
# %%
from annotation_converters.mmdet.cam_box3d import CameraInstance3DBoxes
img_name = "images/chair_batch-17_26_200.jpg"
pred_box = torch.tensor([[-0.0763, -0.0329,  1.2554,  0.4798,  0.6852,  0.5282, -1.5526,  0.0550, 0.1446]])
gt_box = torch.tensor([[-0.0724648179660079, -0.014986128306190949, 1.2463615364915768, 0.47999998927116394, 0.8199999928474426, 0.5199999809265137, -2.0329338055663366, 0.6043210402450145, 0.8726565424208736]])
cam_int = np.array([[730.944091796875, 0.0, 361.7164001464844], [0.0, 730.944091796875, 474.9633483886719], [0.0, 0.0, 1.0]])
new_run = torch.tensor([[-0.0830, -0.0412,  1.3477,  0.4835,  0.6957,  0.5252,  1.5547,  0.0722, 0.0530]])

#%% 
cam_int = np.array([[365.4720458984375, 0.0, 180.8582000732422], [0.0, 365.4720458984375, 237.48167419433594], [0.0, 0.0, 1.0]])
div4 = torch.tensor([[0.3949638903141022, 0.5012848377227783, -1.4408092498779297, 0.46940678358078003, 0.7951834201812744, 0.5258340835571289, -1.390244960784912, 0.3667267858982086, 0.43501797318458557]])
div4 = torch.tensor([[-0.5707, -0.6961,  2.0241,  0.5074,  0.8524,  0.5246, -1.3573, -0.8116, 1.4261]])
div4_single = torch.tensor([[-0.9568, -1.1402,  3.4258,  0.6105,  0.9973,  0.4652,  0.3681, -0.8035,
          0.1539]])
input_box = div4_single
# %% 1
# gt_bboxes = np.concatenate([object_translations[0], object_scales[0], object_euler_angles])
# gt_bboxes = torch.tensor([gt_bboxes])
input_box = gt_box

#%% 2
input_box = pred_box

#%% 3
# use gt rotation to check how rot dimensionality behaves
pred_dim_tensor = pred_box
pred_dim_tensor[:, 6:] = gt_box[:, 6:]
input_box = pred_dim_tensor

#%% exchange location
input_box = torch.tensor([[-0.0725, -0.0150,  1.2464, 0.5074,  0.8524,  0.5246, -2.0329,  0.6043,
          0.8727]])

#%%
pred_yaw_tensor = gt_box
pred_yaw_tensor[:, 7] = pred_box[:, 7]
input_box = pred_yaw_tensor

#%%
pred_yaw_tensor = gt_box
pred_yaw_tensor[:, 8] = pred_box[:, 8]
input_box = pred_yaw_tensor

#%%
pred_yaw_tensor = gt_box
pred_yaw_tensor[:, 6] = pred_box[:, 6]
input_box = pred_yaw_tensor

#%%
#only_rot = torch.tensor([0,0,0])

#input_box = gt_box
gt_bboxes = CameraInstance3DBoxes(tensor=input_box, box_dim=input_box.shape[-1], origin=(0.5,0.5,0.5))
gt_bboxes.corners

h, w, _ = frame[index].shape
image = cv.resize(frame[index].copy(), (w//res_divisor, h//res_divisor))
#image = frame[index].copy()
viz = draw_camera_bbox3d_on_img(
                gt_bboxes, image, cam_int, None, color=(61, 102, 255), thickness=2)

plt.figure(figsize=(10,10))
 
imgplot = plt.imshow(viz)
# %%
diff = gt_box - pred_box 
diff[:, -3:]
# %% "images/chair_batch-17_26_200.jpg"
pred_1 = np.array([[-0.047899723052978516, -0.028579294681549072, 1.2309436798095703, 0.477117121219635, 0.6637790203094482, 0.5223187804222107, -1.512211799621582, 0.06592752784490585, 0.19870617985725403]])
gt_1 = np.array([[-0.05213651026554089, -0.007955606308996721, 1.1857170315877283, 0.47999998927116394, 0.8199999928474426, 0.5199999809265137, -2.227135244771226, -0.45518524191655074, -0.2742213228230022]])
diff_1 = pred_1 - gt_1
diff_1[:, -3:]
# %% "images/chair_batch-46_2_0.jpg"
pred_2 = np.array([[0.09231728315353394, -0.060715436935424805, 1.3331260681152344, 0.530863344669342, 0.7232237458229065, 0.4828373193740845, -1.555285930633545, -0.07119081169366837, 0.2542893886566162]])
gt_2 = np.array([[0.09337218170771, -0.04328000022202616, 1.363566804221949, 0.47999995946884155, 0.9699999690055847, 0.5099999904632568, -2.4991929837289972, -0.31979073068841246, -0.22552889779232663]])
diff_2 = pred_2 - gt_2
diff_2[:, -3:]

#%% "images/chair_batch-31_45_0.jpg"
pred_3 = np.array([[-0.04115968942642212, 0.030712127685546875, 1.3547592163085938, 0.5089847445487976, 0.713020920753479, 0.4886595904827118, -1.5180573463439941, 0.04662647843360901, 0.09937775135040283]])
gt_3 = np.array([[-0.03807137904631652, 0.04572881338853407, 1.2529250907648253, 0.5199999809265137, 0.699999988079071, 0.5, -2.182419140080946, 0.7015109738920131, 0.6356461819784811]])
diff_3 = pred_3 - gt_3
diff_3[:, -3:]

# %% img_name = "images/chair_batch-17_26_300.jpg" half resolution
res_divisor = 2
cam_int = np.array([[732.0821533203125, 0.0, 360.9198913574219], [0.0, 732.0821533203125, 476.89471435546875], [0.0, 0.0, 1.0]])
pred_box = torch.tensor([[-0.0050,  0.0590,  0.8594,  0.7315,  1.0873,  0.5957,  1.3208, -0.5107, -0.4394]])
gt_box = torch.tensor([[0.002357743083003072, 0.07448509794829761, 1.072909751111073, 0.47999998927116394, 0.8199999928474426, 0.5199999809265137, -2.0750782722804204, 0.11210464114280283, 0.14604112103076475]])
pred_box_x = torch.tensor([[-2.3144e-04,  7.9374e-02,  1.0707e+00,  7.5078e-01,  1.0477e+00,
          5.4066e-01, -4.2729e+00, -3.5146e-01, -6.8672e-01]])
# %%
pred_box - gt_box
# %%

# %%

# %%
cam_int = np.array([[730.944091796875, 0.0, 361.7164001464844], [0.0, 730.944091796875, 474.9633483886719], [0.0, 0.0, 1.0]])
input_box = torch.tensor([[0.002357743083003072, 0.07448509794829761, 1.072909751111073, 0.47999998927116394, 0.8199999928474426, 0.5199999809265137, -2.0750782722804204, 0.11210464114280283, 0.14604112103076475]])
pred_box = torch.tensor([[ 0.0025,  0.0664,  1.1699,  0.4807,  0.8199,  0.5203, -2.0878,  0.1129,
          0.1517]])

#%%
cam_int = np.array([[790.5274047851562, 0.0, 355.4814453125], [0.0, 790.5274047851562, 469.391845703125], [0.0, 0.0, 1.0]])
input_box = torch.tensor([[0.015977288545049984, -0.14467811737139513, 1.5282741925131376, 0.5299999713897705, 0.7400000095367432, 0.5600000023841858, -1.8939632471951076, 0.9447955322767987, 1.2039131998234998]])
pred_box = torch.tensor([[ 0.0156, -0.1388,  1.4944,  0.5308,  0.7394,  0.5593,  0.7759, -1.2562,
         -0.7649]])
pred_box = torch.tensor([[ 0.0161, -0.1463,  1.5714,  0.5304,  0.7402,  0.5601, -0.2904, -0.0248,
         -1.3016]])

# %%
#input_box = torch.tensor([[0.015977288545049984, -0.14467811737139513, 1.5282741925131376, 0.5299999713897705, 0.7400000095367432, 0.5600000023841858, 0, 0, 0]])
input_box = torch.tensor([[0.015977288545049984, -0.14467811737139513, 1.5282741925131376, 0.5299999713897705, 0.7400000095367432, 0.5600000023841858, 0 , 0.9447955322767987, 0 ]]) #y
input_box = torch.tensor([[0.015977288545049984, -0.14467811737139513, 1.5282741925131376, 0.5299999713897705, 0.7400000095367432, 0.5600000023841858, -1.8939632471951076, 0.9447955322767987 , 0 ]]) #x #y
input_box = torch.tensor([[0.015977288545049984, -0.14467811737139513, 1.5282741925131376, 0.5299999713897705, 0.7400000095367432, 0.5600000023841858, 0, 0, 1.2039131998234998]])

#%%
cam_int = np.array([[732.0821533203125, 0.0, 360.9198913574219], [0.0, 732.0821533203125, 476.89471435546875], [0.0, 0.0, 1.0]])
# pred_box = torch.tensor([[-0.0050,  0.0590,  0.8594,  0.7315,  1.0873,  0.5957,  1.3208, -0.5107, -0.4394]])
input_box = torch.tensor([[0.002357743083003072, 0.07448509794829761, 1.072909751111073, 0.47999998927116394, 0.8199999928474426, 0.5199999809265137, -2.0750782722804204, 0.11210464114280283, 0.14604112103076475]])

#%%
gt_bboxes = CameraInstance3DBoxes(tensor=input_box, box_dim=input_box.shape[-1], origin=(0.5,0.5,0.5))
pred_bboxes = CameraInstance3DBoxes(tensor=pred_box, box_dim=input_box.shape[-1], origin=(0.5,0.5,0.5))

h, w, _ = frame[index].shape
image = cv.resize(frame[index].copy(), (w//res_divisor, h//res_divisor))
#image = frame[index].copy()
# viz = draw_camera_bbox3d_on_img(
#                 pred_bboxes, image, cam_int, None, color=(61, 102, 255), thickness=2)
viz2 = draw_camera_bbox3d_on_img(
                gt_bboxes, image, cam_int, None, color=(255, 102, 61), thickness=2)
plt.figure(figsize=(10,10))
 
imgplot = plt.imshow(viz2)

#%% flip image test
h, w, _ = frame[index].shape
#input_box[0, 7] -= np.pi
gt_bboxes = CameraInstance3DBoxes(tensor=input_box, box_dim=input_box.shape[-1], origin=(0.5,0.5,0.5))
#input_box[0][6:9] = torch.tensor([np.pi/4, 0, 0]).float()
gt_bboxes = CameraInstance3DBoxes(tensor=input_box, box_dim=input_box.shape[-1], origin=(0.5,0.5,0.5))
gt_bboxes.flip()
originalImage = cv.resize(frame[index].copy(), (w//res_divisor, h//res_divisor))
flipHorizontal = cv2.flip(originalImage, 1)
viz = draw_camera_bbox3d_on_img(
                gt_bboxes, flipHorizontal, cam_int, None, color=(61, 102, 255), thickness=2)


plt.figure(figsize=(10,10))
imgplot = plt.imshow(viz)

#%%
rot = np.array([
                [0, 1, 0], 
                [1, 0, 0], 
                [0, 0, -1]])

# %%

rot_adjustment = np.array([
                              [0, 1, 0], 
                              [1, 0, 0], 
                              [0, 0, -1]])
trafo_2_obj = np.eye(4)
trafo_2_obj[:3, :3] = rot_adjustment

def convert(result, coco_label, intrinsics):
    """
      Implement your own function/model to predict the box's 2D and 3D 
      keypoint from the input images. 
      Note that the predicted 3D bounding boxes are correct upto an scale. 
      You can use the ground planes to re-scale your boxes if necessary. 

      Returns: TODO: Just a list of boxes
        A list of list of boxes for objects in images in the batch. Each box is 
        a tuple of (point_2d, point_3d) that includes the predicted 2D and 3D vertices.
    """

    # coco_label:
    # dict of numpy arrays: several bbox tensors in 2d np array encoded
    # ann_info
                    # ann = dict(
                    # bboxes=gt_bboxes,
                    # labels=gt_labels,
                    # gt_bboxes_3d=gt_bboxes_cam3d,
                    # gt_labels_3d=gt_labels_3d,
                    # centers2d=centers2d,
                    # depths=depths,
                    # bboxes_ignore=gt_bboxes_ignore,
                    # masks=gt_masks_ann,
                    # seg_map=seg_map)
            # img_info: intrinsics

    # result is a single dict: bbox, box3d_camera, scores, label_preds, sample_idx
    # each contains a list of respective objects

    # alles 2d tensoren or arrays

    # conversion between mmdetection coordinate system and objectron convention
    # negate z and swap x & y in points

    # results conversion
    results_converted = []
    labels_converted = []

    for box_tensor in result.tensor: # box3d_camera if convert_valid_boxes called before

      box_tensor = box_tensor.numpy() 
      # rotation, translation, scale
      box = Box.from_transformation(box_tensor[6:], box_tensor[:3], box_tensor[3:6]) 
      # project 9 keypoints to 2D
      vertices_2d = points_cam2img(torch.tensor(box.vertices).float(), intrinsics)
      vertices_2d[:, 0] /= 720
      vertices_2d[:, 1] /= 960

      # transform box back to objectron convention
      obj_box = box.apply_transformation(trafo_2_obj)
      # combine to output format: tuple of (point_2d, point_3d)
      # devide keypoints by 255 to normalize them
      results_converted.append((vertices_2d.numpy(), obj_box.vertices))

    # gt conversion: coco 2 objectron

    for box_tensor in coco_label.tensor:
      
      box_tensor = box_tensor.numpy() 
      # rotation, translation, scale
      box = Box.from_transformation(box_tensor[6:], box_tensor[:3], box_tensor[3:6]) 
      # project 9 keypoints to 2D
      vertices_2d = points_cam2img(torch.tensor(box.vertices).float(), intrinsics)
      vertices_2d[:, 0] /= 720
      vertices_2d[:, 1] /= 960

      # transform box back to objectron convention
      obj_box = box.apply_transformation(trafo_2_obj)
      # combine to output format: tuple of (point_2d, point_3d)
      # devide keypoints by 255 to normalize them
      labels_converted.append((vertices_2d.numpy(), obj_box.vertices))

    return results_converted, labels_converted

results_converted, labels_converted = convert(pred_bboxes, gt_bboxes, torch.tensor(cam_int).float())
# %%
from objectron.dataset.iou import IoU
def evaluate_iou(box, instance):
    """Evaluates a 3D box by 3D IoU.

    It computes 3D IoU of predicted and annotated boxes.

    Args:
      box: A 9*3 array of a predicted box.
      instance: A 9*3 array of an annotated box, in metric level.

    Returns:
      3D Intersection over Union (float)
    """
    # Computes 3D IoU of the two boxes.
    prediction = Box(box)
    annotation = Box(instance)
    iou = IoU(prediction, annotation)
    iou_result = iou.iou()
    #self._iou_3d += iou_result
    print('IOU sample: ', iou_result)
    return iou_result

evaluate_iou(results_converted[0][1],labels_converted[0][1])

#%%
evaluate_iou(results_converted[0][1],results_converted[0][1] + np.array([0.01, 0.02, 0.5]))

#%%
evaluate_iou(results_converted[0][1],results_converted[0][1] + np.array([0, 0, 0.09]))

# %%

image = draw_annotation_on_image(frame[index].copy(), results_converted[0][0], [9]) # test_box_zr.vertices
image_2 = draw_annotation_on_image(image, labels_converted[0][0], [9]) # test_box_zr.vertices
plt.figure(figsize=(10,10))
imgplot = plt.imshow(image_2)

# %%
diff_obj = results_converted[0][1] - labels_converted[0][1]
diff_obj[1:]
#%%
diff = pred_bboxes.corners - gt_bboxes.corners
diff
# %% chair_batch-30_3_250.jpg

gt = np.array([0.04661225171860828, -0.05867959404303846, 1.3754354040416743, 0.5099999904632568, 0.7099999785423279, 0.47999998927116394, -1.318476788945011, -0.8228192795455149, -2.0207379502602323])
pred = np.array([ 0.0595, -0.0406,  1.3535,  0.5141,  0.7376,  0.5276, -2.6511, -0.6164,
         -1.2998])
gt - pred
# %%
from scipy.spatial.transform import Rotation as R
rottt = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
r = R.from_matrix(rottt)
# what rotation representation do I need?
object_euler_angles = r.as_euler('xyz', degrees=False)
print(r.as_euler('xyz', degrees=True))

# %%
input = R.from_euler('xyz', np.array([np.pi/4, np.pi/4, np.pi/4]))

R.from_matrix(np.dot(rottt, input.as_matrix())).as_euler('xyz', degrees=False) - np.array([np.pi/4, np.pi/4, np.pi/4])
# %%
data = mmcv.load("objectron_processed_chair_all/annotations/objectron_train.json")

wrong_samples = {}
for sample in data['annotations']:
    if abs(sample["bbox_cam3d"][2])>100:
        print(sample["bbox_cam3d"])
        print(sample['file_name'])
        wrong_samples[sample['file_name'][:-7]] = 1
wrong_samples

# %%
cam_int = np.array([[392.91546630859375, 0.0, 177.7733154296875], [0.0, 392.91546630859375, 234.90794372558594], [0.0, 0.0, 1.0]])
bbox = torch.tensor([[-0.0038461225112547126, -0.06066409704894937, 1.4329994183863732, 0.5299999713897705, 0.7400000095367432, 0.5600000023841858, -1.4534226265612311, 1.0372450503471806, 1.6832464514912657]])
res_divisor = 4

#%%
cam_int = np.array([[392.91143798828125, 0.0, 180.07415771484375], [0.0, 392.91143798828125, 235.15142822265625], [0.0, 0.0, 1.0]])
bbox = torch.tensor([[0.0016107390292674495, 0.02712655255045071, 1.739724733980708, 0.4692857265472412, 0.9392856955528259, 0.5344047546386719, -2.335701110051221, -0.7985773581267579, -0.7583603966889692]])
res_divisor = 4
# %%

gt_bboxes = CameraInstance3DBoxes(tensor=bbox, box_dim=bbox.shape[-1], origin=(0.5,0.5,0.5))

h, w, _ = frame[index].shape
image = cv.resize(frame[index].copy(), (w//res_divisor, h//res_divisor))
#image = frame[index].copy()
viz = draw_camera_bbox3d_on_img(
                gt_bboxes, image, cam_int, None, color=(61, 102, 255), thickness=2)

plt.figure(figsize=(10,10))
 
imgplot = plt.imshow(viz)

# %% Annotations
import mmcv
annotations = {
    "images": [],
    "categories": [{"id": 0, "name": "chair", "id": 1, "name": "book"}]
}

for i in range(3000):
    annotations["images"] += [{"file_name": "output_{:05d}.png".format(i+1), "id": i, "cam_intrinsic": [[1450, 0.0, 932], [0.0, 1450, 720], [0.0, 0.0, 1.0]]}]
    
mmcv.dump(annotations, 'export_annotations.json')
# %%
