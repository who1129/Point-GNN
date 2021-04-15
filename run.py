"""This file implement an inference pipeline for Point-GNN on KITTI dataset"""

import os
import time
import argparse
import multiprocessing
from functools import partial

import numpy as np
import tensorflow as tf
import open3d
import cv2
from tqdm import tqdm

from dataset.kitti_dataset import KittiDataset, Points
from models.graph_gen import get_graph_generate_fn
from models.models import get_model
from models.box_encoding import get_box_decoding_fn, get_box_encoding_fn, \
                          get_encoding_len
from models import preprocess
from models import nms
from util.config_util import load_config, load_train_config
from util.summary_util import write_summary_scale

parser = argparse.ArgumentParser(description='Point-GNN inference on KITTI')
parser.add_argument('checkpoint_path', type=str,
                   help='Path to checkpoint')
parser.add_argument('--test', dest='test', action='store_true',
                    default=False, help='Enable test model')
parser.add_argument('--no-box-merge', dest='use_box_merge',
                    action='store_false', default='True',
                   help='Disable box merge.')
parser.add_argument('--no-box-score', dest='use_box_score',
                    action='store_false', default='True',
                   help='Disable box score.')
parser.add_argument('--dataset_root_dir', type=str, default='../dataset/kitti/',
                   help='Path to KITTI dataset. Default="../dataset/kitti/"')
parser.add_argument('--dataset_split_file', type=str,
                    default='',
                   help='Path to KITTI dataset split file.'
                   'Default="DATASET_ROOT_DIR/3DOP_splits/val.txt"')
parser.add_argument('--output_dir', type=str,
                    default='',
                   help='Path to save the detection results'
                   'Default="CHECKPOINT_PATH/eval/"')
args = parser.parse_args()
IS_TEST = args.test
USE_BOX_MERGE = args.use_box_merge
USE_BOX_SCORE = args.use_box_score
DATASET_DIR = args.dataset_root_dir
if args.dataset_split_file == '':
    DATASET_SPLIT_FILE = os.path.join(DATASET_DIR, './3DOP_splits/val.txt')
else:
    DATASET_SPLIT_FILE = args.dataset_split_file
if args.output_dir == '':
    OUTPUT_DIR = os.path.join(args.checkpoint_path, './eval/')
else:
    OUTPUT_DIR = args.output_dir
CHECKPOINT_PATH = args.checkpoint_path
CONFIG_PATH = os.path.join(CHECKPOINT_PATH, 'config')
assert os.path.isfile(CONFIG_PATH), 'No config file found in %s'
config = load_config(CONFIG_PATH)
# setup dataset ===============================================================
if IS_TEST:
    dataset = KittiDataset(
        os.path.join(DATASET_DIR, 'image/testing/image_2'),
        os.path.join(DATASET_DIR, 'velodyne/testing/'),
        os.path.join(DATASET_DIR, 'calib/testing/calib'),
        '',
        num_classes=config['num_classes'],
        is_training=False)
else:
    dataset = KittiDataset(
        os.path.join(DATASET_DIR, 'image/training/image_2'),
        os.path.join(DATASET_DIR, 'velodyne/training/'),
        os.path.join(DATASET_DIR, 'calib/training/calib'),
        os.path.join(DATASET_DIR, 'labels/training'),
        DATASET_SPLIT_FILE,
        num_classes=config['num_classes'])
NUM_TEST_SAMPLE = dataset.num_files
NUM_CLASSES = dataset.num_classes
try:
    for frame_idx in tqdm(range(0, NUM_TEST_SAMPLE)):
        start_time = time.time()
        # provide input ======================================================
        cam_points = dataset.get_cam_points_in_image(frame_idx,
            config['downsample_by_voxel_size'])
        
except:
    print("ss")



# occlusion score =============================================================
def occlusion(label, xyz):
    if xyz.shape[0] == 0:
        return 0
    normals, lower, upper = dataset.box3d_to_normals(label)
    projected = np.matmul(xyz, np.transpose(normals))
    x_cover_rate = (np.max(projected[:, 0])-np.min(projected[:, 0]))\
        /(upper[0] - lower[0])
    y_cover_rate = (np.max(projected[:, 1])-np.min(projected[:, 1]))\
        /(upper[1] - lower[1])
    z_cover_rate = (np.max(projected[:, 2])-np.min(projected[:, 2]))\
        /(upper[2] - lower[2])
    return x_cover_rate*y_cover_rate*z_cover_rate
# setup model =================================================================
BOX_ENCODING_LEN = get_encoding_len(config['box_encoding_method'])
box_encoding_fn = get_box_encoding_fn(config['box_encoding_method'])
box_decoding_fn = get_box_decoding_fn(config['box_encoding_method'])
t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 1])
t_vertex_coord_list = [tf.placeholder(dtype=tf.float32, shape=[None, 3])]
for _ in range(len(config['runtime_graph_gen_kwargs']['level_configs'])):
    t_vertex_coord_list.append(
        tf.placeholder(dtype=tf.float32, shape=[None, 3]))
t_edges_list = []
for _ in range(len(config['runtime_graph_gen_kwargs']['level_configs'])):
    t_edges_list.append(
        tf.placeholder(dtype=tf.int32, shape=[None, 2]))
t_keypoint_indices_list = []
for _ in range(len(config['runtime_graph_gen_kwargs']['level_configs'])):
    t_keypoint_indices_list.append(
        tf.placeholder(dtype=tf.int32, shape=[None, 1]))
t_is_training = tf.placeholder(dtype=tf.bool, shape=[])
model = get_model(config['model_name'])(num_classes=NUM_CLASSES,
    box_encoding_len=BOX_ENCODING_LEN, mode='test', **config['model_kwargs'])
t_logits, t_pred_box = model.predict(
    t_initial_vertex_features, t_vertex_coord_list, t_keypoint_indices_list,
    t_edges_list,
    t_is_training)
t_probs = model.postprocess(t_logits)
t_predictions = tf.argmax(t_probs, axis=1, output_type=tf.int32)
# optimizers ==================================================================
global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
fetches = {
    'step': global_step,
    'predictions': t_predictions,
    'probs': t_probs,
    'pred_box': t_pred_box
    }
# runing network ==============================================================
print("runing network....")
from tensorflow.python.client import device_lib

print("gpu: ",device_lib.list_local_devices())
print(dataset)
time_dict = {}
saver = tf.train.Saver()
graph = tf.get_default_graph()
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(graph=graph,
    config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.variables_initializer(tf.global_variables()))
    sess.run(tf.variables_initializer(tf.local_variables()))
    model_path = tf.train.latest_checkpoint(CHECKPOINT_PATH)
    print('Restore from checkpoint %s' % model_path)
    saver.restore(sess, model_path)
    previous_step = sess.run(global_step)
    for frame_idx in tqdm(range(0, NUM_TEST_SAMPLE)):
        start_time = time.time()
        # provide input ======================================================
        cam_points = dataset.get_cam_points_in_image(frame_idx,
            config['downsample_by_voxel_size'])
        
        calib = dataset.get_calib(frame_idx)
        
        if not IS_TEST:
            box_label_list = dataset.get_label(frame_idx)
        input_time = time.time()
        time_dict['fetch input'] = time_dict.get('fetch input', 0) \
            + input_time - start_time
        graph_generate_fn= get_graph_generate_fn(config['graph_gen_method'])
        (vertex_coord_list, keypoint_indices_list, edges_list) = \
            graph_generate_fn(
                cam_points.xyz, **config['runtime_graph_gen_kwargs'])
        graph_time = time.time()
        time_dict['gen graph'] = time_dict.get('gen graph', 0) \
            + graph_time - input_time
        input_v = cam_points.attr
        
        last_layer_graph_level = \
            config['model_kwargs']['layer_configs'][-1]['graph_level']
        last_layer_points_xyz = vertex_coord_list[last_layer_graph_level+1]
        if config['label_method'] == 'yaw':
            label_map = {'Background': 0, 'Car': 1, 'Pedestrian': 3,
                'Cyclist': 5,'DontCare': 7}
        if config['label_method'] == 'Car':
            label_map = {'Background': 0, 'Car': 1, 'DontCare': 3}
        if config['label_method'] == 'Pedestrian_and_Cyclist':
            label_map = {'Background': 0, 'Pedestrian': 1, 'Cyclist':3,
                'DontCare': 5}
        # run forwarding =====================================================
        feed_dict = {
            t_initial_vertex_features: input_v,
            t_is_training: True,
        }
        feed_dict.update(dict(zip(t_edges_list, edges_list)))
        feed_dict.update(
            dict(zip(t_keypoint_indices_list, keypoint_indices_list)))
        feed_dict.update(dict(zip(t_vertex_coord_list, vertex_coord_list)))
        results = sess.run(fetches, feed_dict=feed_dict)
        gnn_time = time.time()
        time_dict['gnn inference'] = time_dict.get('gnn inference', 0) \
            + gnn_time - graph_time
        # box decoding =======================================================
        box_probs = results['probs']
        box_labels = np.tile(np.expand_dims(np.arange(NUM_CLASSES), axis=0),
            (box_probs.shape[0], 1))
        box_labels = box_labels.reshape((-1))
        raw_box_labels = box_labels
        box_probs = box_probs.reshape((-1))
        pred_boxes = results['pred_box'].reshape((-1, 1, BOX_ENCODING_LEN))
        last_layer_points_xyz = np.tile(
            np.expand_dims(last_layer_points_xyz, axis=1), (1, NUM_CLASSES, 1))
        last_layer_points_xyz = last_layer_points_xyz.reshape((-1, 3))
        boxes_centers = last_layer_points_xyz
        decoded_boxes = box_decoding_fn(np.expand_dims(box_labels, axis=1),
            boxes_centers, pred_boxes, label_map)
        box_mask = (box_labels > 0)*(box_labels < NUM_CLASSES-1)
        box_mask = box_mask*(box_probs > 1./NUM_CLASSES)
        box_indices = np.nonzero(box_mask)[0]
        decode_time = time.time()
        time_dict['decode box'] = time_dict.get('decode box', 0) \
            + decode_time - gnn_time
        if box_indices.size != 0:
            box_labels = box_labels[box_indices]
            box_probs = box_probs[box_indices]
            box_probs_ori = box_probs
            decoded_boxes = decoded_boxes[box_indices, 0]
            box_labels[box_labels==2]=1
            box_labels[box_labels==4]=3
            box_labels[box_labels==6]=5
            detection_scores = box_probs
            # nms ============================================================
            if USE_BOX_MERGE and USE_BOX_SCORE:
                (class_labels, detection_boxes_3d, detection_scores,
                nms_indices) = nms.nms_boxes_3d_uncertainty(
                    box_labels, decoded_boxes, detection_scores,
                    overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                    overlapped_thres=config['nms_overlapped_thres'],
                    appr_factor=100.0, top_k=-1,
                    attributes=np.arange(len(box_indices)))
            if USE_BOX_MERGE and not USE_BOX_SCORE:
                (class_labels, detection_boxes_3d, detection_scores,
                nms_indices) = nms.nms_boxes_3d_merge_only(
                    box_labels, decoded_boxes, detection_scores,
                    overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                    overlapped_thres=config['nms_overlapped_thres'],
                    appr_factor=100.0, top_k=-1,
                    attributes=np.arange(len(box_indices)))
            if not USE_BOX_MERGE and USE_BOX_SCORE:
                (class_labels, detection_boxes_3d, detection_scores,
                nms_indices) = nms.nms_boxes_3d_score_only(
                    box_labels, decoded_boxes, detection_scores,
                    overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                    overlapped_thres=config['nms_overlapped_thres'],
                    appr_factor=100.0, top_k=-1,
                    attributes=np.arange(len(box_indices)))
            if not USE_BOX_MERGE and not USE_BOX_SCORE:
                (class_labels, detection_boxes_3d, detection_scores,
                nms_indices) = nms.nms_boxes_3d(
                    box_labels, decoded_boxes, detection_scores,
                    overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                    overlapped_thres=config['nms_overlapped_thres'],
                    appr_factor=100.0, top_k=-1,
                    attributes=np.arange(len(box_indices)))
            box_probs = detection_scores
            
            # convert to KITTI ================================================
            detection_boxes_3d_corners = nms.boxes_3d_to_corners(
                detection_boxes_3d)
            pred_labels = []
            for i in range(len(detection_boxes_3d_corners)):
                detection_box_3d_corners = detection_boxes_3d_corners[i]
                corners_cam_points = Points(
                    xyz=detection_box_3d_corners, attr=None)
                corners_img_points = dataset.cam_points_to_image(
                    corners_cam_points, calib)
                corners_xy = corners_img_points.xyz[:, :2]
                if config['label_method'] == 'yaw':
                    all_class_name = ['Background', 'Car', 'Car', 'Pedestrian',
                        'Pedestrian', 'Cyclist', 'Cyclist', 'DontCare']
                if config['label_method'] == 'Car':
                    all_class_name = ['Background', 'Car', 'Car', 'DontCare']
                if config['label_method'] == 'Pedestrian_and_Cyclist':
                    all_class_name = ['Background', 'Pedestrian', 'Pedestrian',
                        'Cyclist', 'Cyclist', 'DontCare']
                if config['label_method'] == 'alpha':
                    all_class_name = ['Background', 'Car', 'Car', 'Pedestrian',
                        'Pedestrian', 'Cyclist', 'Cyclist', 'DontCare']
                class_name = all_class_name[class_labels[i]]
                xmin, ymin = np.amin(corners_xy, axis=0)
                xmax, ymax = np.amax(corners_xy, axis=0)
                clip_xmin = max(xmin, 0.0)
                clip_ymin = max(ymin, 0.0)
                clip_xmax = min(xmax, 1242.0)
                clip_ymax = min(ymax, 375.0)
                height = clip_ymax - clip_ymin
                truncation_rate = 1.0 - (clip_ymax - clip_ymin)*(
                    clip_xmax - clip_xmin)/((ymax - ymin)*(xmax - xmin))
                if truncation_rate > 0.4:
                    continue
                x3d, y3d, z3d, l, h, w, yaw = detection_boxes_3d[i]
                assert l > 0, str(i)
                score = box_probs[i]
                if USE_BOX_SCORE:
                    tmp_label = {"x3d": x3d, "y3d" : y3d, "z3d": z3d,
                    "yaw": yaw, "height": h, "width": w, "length": l}
                    # Rescore or not ===========================================
                    inside_mask = dataset.sel_xyz_in_box3d(tmp_label,
                        last_layer_points_xyz[box_indices])
                    points_inside = last_layer_points_xyz[
                        box_indices][inside_mask]
                    score_inside = box_probs_ori[inside_mask]
                    score = (1+occlusion(tmp_label, points_inside))*score
                pred_labels.append((class_name, -1, -1, 0,
                    clip_xmin, clip_ymin, clip_xmax, clip_ymax,
                    h, w, l, x3d, y3d, z3d, yaw, score))
                
            nms_time = time.time()
            time_dict['nms'] = time_dict.get('nms', 0) + nms_time - decode_time
            # output ===========================================================
            filename = OUTPUT_DIR+'/data/'+dataset.get_filename(
                frame_idx)+'.txt'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                for pred_label in pred_labels:
                    for field in pred_label:
                        f.write(str(field)+' ')
                    f.write('\n')
                f.write('\n')
            
        else:
            filename = OUTPUT_DIR+'/data/'+dataset.get_filename(
                frame_idx)+'.txt'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                f.write('\n')

        total_time = time.time()
        time_dict['total'] = time_dict.get('total', 0) \
            + total_time - start_time
    for key in time_dict:
        print(key + " time : " + str(time_dict[key]/NUM_TEST_SAMPLE))
