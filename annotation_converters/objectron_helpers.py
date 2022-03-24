import os
import sys
import subprocess

import cv2
import numpy as np

module_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../3rdparty/Objectron'))
if module_path not in sys.path:
    sys.path.append(module_path)

#pylint: disable = wrong-import-position
from objectron.schema import annotation_data_pb2 as annotation_protocol

# The annotations are stored in protocol buffer format.
# The AR Metadata captured with each frame in the video

from scipy.spatial.transform import Rotation as R

def get_frame_annotation(sequence, frame_id):
    """Grab an annotated frame from the sequence."""
    data = sequence.frame_annotations[frame_id]
    object_id = 0
    object_keypoints_2d = []
    object_keypoints_3d = []
    #object_rotations = []
    #object_translations = []
    #object_scale = []
    num_keypoints_per_object = []
    object_categories = []
    annotation_types = []
    object_bboxes_3d = []
    #visibilities = []

    # Get the camera for the current frame. We will use the camera to bring
    # the object from the world coordinate to the current camera coordinate.
    #camera = np.array(data.camera.transform).reshape(4, 4)
    view_matrix = np.array(data.camera.view_matrix).reshape(4, 4)
    intrinsics = np.array(data.camera.intrinsics).reshape(3, 3)

    # adjustment of coordinate system conventions:
    # - convert from objectron into mmdetection coordinate system
    # - negate z and swap x & y in points
    # - sawp px & py in intrinsics

    intrinsics[0,2], intrinsics[1,2] = intrinsics[1,2], intrinsics[0,2]

    for obj in sequence.objects: # what happens to objects not visible in the scene?
        rotation = np.array(obj.rotation).reshape(3, 3)

        translation = np.array(obj.translation)
        # scale invariant
        # object_scale.append(np.array(obj.scale))
        transformation = np.identity(4)
        transformation[:3, :3] = rotation
        transformation[:3, 3] = translation
        # transformation into objectron camera coordinates
        obj_cam = np.matmul(view_matrix, transformation)

        # convert into mmdetection coordinate system
        # negate z and swap x & y in points
        rot_adjustment = np.array([
            [0, 1, 0], 
            [1, 0, 0], 
            [0, 0, -1]])
        transformation_adj = np.eye(4)
        transformation_adj[:3, :3] = rot_adjustment
        obj_cam = np.matmul(transformation_adj, obj_cam)

        # object_translations.append(obj_cam[:3, 3])
        # object_rotations.append(obj_cam[:3, :3])

        # transfer rotation matrix to euler angles
        object_categories.append(obj.category)
        annotation_types.append(obj.type)
        # unit_points.append(obj.keypoints) # contains box vertices in the "BOX" coordinate, (i.e. it's a unit box)

        # we need translation, scale & all three rotations in euler angles
        trans = obj_cam[:3, 3].tolist() 
        scale = list(obj.scale)
        rot = R.from_matrix(obj_cam[:3, :3])
        euler_angles = rot.as_euler('xyz', degrees=False).tolist()

        bbox_3d = trans + scale + euler_angles
        object_bboxes_3d.append(bbox_3d)

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
        #visibilities.append(annotations.visibility)
    return [object_keypoints_2d, object_categories, keypoint_size_list,
            annotation_types, object_bboxes_3d, intrinsics.tolist(), view_matrix.tolist()] #, object_keypoints_3d, visibilities]


def get_video_frames_number(video_file):
    capture = cv2.VideoCapture(video_file)
    return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))


def grab_frames(video_file, frame_ids, use_opencv=True):
    """Grab an image frame from the video file."""
    frames = {}
    capture = cv2.VideoCapture(video_file)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    # totalFrames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if use_opencv:
        for frame_id in frame_ids:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            _, current_frame = capture.read()
            frames[frame_id] = current_frame
        capture.release()
    else:
        frame_size = width * height * 3

        for frame_id in frame_ids:
            frame_filter = r'select=\'eq(n\,{:d})\''.format(frame_id)
            command = [
                'ffmpeg', '-i', video_file, '-f', 'image2pipe', '-vf', frame_filter,
                '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-vsync', 'vfr', '-'
            ]
            with subprocess.Popen(
                command, stdout=subprocess.PIPE, bufsize=151 * frame_size, stderr=subprocess.DEVNULL) as pipe:
                current_frame = np.frombuffer(
                    pipe.stdout.read(frame_size), dtype='uint8').reshape(height, width, 3)
                pipe.stdout.flush()
                frames[frame_id] = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

    return frames


def load_annotation_sequence(annotation_file):
    frame_annotations = []
    with open(annotation_file, 'rb') as pb:
        sequence = annotation_protocol.Sequence()
        sequence.ParseFromString(pb.read())
        for i in range(len(sequence.frame_annotations)):
            frame_annotations.append(get_frame_annotation(sequence, i))
           # annotation, cat, num_keypoints, types
    return frame_annotations
