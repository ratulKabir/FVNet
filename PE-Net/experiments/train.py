from __future__ import print_function

import os
import sys
import time
import argparse
import importlib
import numpy as np
from datetime import datetime

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # directory of the file: /home/ubuntu/workstation/FVNet/PE-Net/experiments
ROOT_DIR = os.path.dirname(BASE_DIR)    # /home/ubuntu/workstation/FVNet/PE-Net
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'kitti'))

from kitti_dataset import KittiDataset, get_batch
from kitti_util import compute_box3d_iou

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument('--model', default='front_pointnets_v1', help='Model name')
parser.add_argument('--output_dir', default='../outputs', help='Log dir')
parser.add_argument('--num_point', type=int, default=512, help='Point Number')  # ????
parser.add_argument('--max_epoch', type=int, default=1001, help='Epoch to run')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training')
parser.add_argument('--batch_size_eval', type=int, default=64, help='Batch Size during evaluation')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--optimizer', default='adam', help='adam')
parser.add_argument('--decay_step', type=int, default=800000, help='Decay step for lr decay')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay')
parser.add_argument('--restore_model_path', default=None, help='Restore model path e.g. log/model.ckpt')
FLAGS = parser.parse_args()

EPOCH_CNT = 0
MAX_EPOCH = FLAGS.max_epoch
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
BASE_LEARNING_RATE = FLAGS.learning_rate
BATCH_SIZE = FLAGS.batch_size
BATCH_SIZE_EVAL = FLAGS.batch_size_eval
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
NUM_CHANNEL = 4

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model + '.py')

time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)
LOG_DIR = os.path.join(FLAGS.output_dir, time_stamp)
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
os.system('cp -r ../experiments/ ../kitti/ ../models/ ../kitti_eval %s' % (LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

prefix = "/home/ubuntu/workstation/data/dataset/kitti_fvnet2/refinement/" # /home/ratul/data/dataset/kitti_fvnet2/refinement/   
DATA_DIR = prefix + "training"
TRAIN_LIST_FILE = prefix + "list_files/det_train_car_filtered.txt"
TRAIN_LABEL_FILE = prefix + "list_files/label_train_car_filtered.txt"
TRAIN_DATASET = KittiDataset(NUM_POINT, DATA_DIR, TRAIN_LIST_FILE, TRAIN_LABEL_FILE, perturb_box=True, aug=True)

VAL_LIST_FILE = prefix + "list_files/det_val_car_filtered.txt"
VAL_LABEL_FILE = prefix + "list_files/label_val_car_filtered.txt"
VAL_DATASET = KittiDataset(NUM_POINT, DATA_DIR, VAL_LIST_FILE, VAL_LABEL_FILE)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

def train():
    num_points = NUM_POINT
    pe_net = MODEL.Front_PointNet(num_points)
    pe_net.to(DEVICE)

    # Get training operator
    # learning_rate = get_learning_rate(batch)
    # tf.summary.scalar('learning_rate', learning_rate)
    if OPTIMIZER == 'momentum':
        # optimizer = tf.train.MomentumOptimizer(learning_rate,
        #                                        momentum=MOMENTUM)
        pass
    elif OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(pe_net.parameters())
    # train_op = optimizer.minimize(total_loss, global_step=batch)

    for epoch in range(MAX_EPOCH):
        log_string('**** EPOCH %03d ****' % (epoch))
        sys.stdout.flush()

        train_one_epoch(pe_net, optimizer)

        # Save the variables to disk.
        if epoch > 0 and epoch % 5 == 0:
            eval_one_epoch(pe_net)
        if epoch > 0 and epoch % 20 == 0:
            save_path = saver.save(sess, os.path.join(LOG_DIR, "model_" + str(epoch) + ".ckpt"))
            log_string("Model saved in file: %s" % save_path)


def train_one_epoch(pe_net, optimizer):
    is_training = True
    log_string(str(datetime.now()))

    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = int(np.ceil(len(TRAIN_DATASET) / BATCH_SIZE))

    # To collect statistics

    total_loss_sum = 0
    center_loss_sum = 0
    stage1_center_loss_sum = 0
    h_cls_loss_sum = 0
    h_res_loss_sum = 0
    s_res_loss_sum = 0
    corners_loss_sum = 0

    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt_50 = 0
    iou3d_correct_cnt_70 = 0

    # Training with batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        batch_data = get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx, NUM_CHANNEL)

        batch_data, batch_center, \
        batch_angle_cls, batch_angle_res, batch_size_res = [torch.from_numpy(data).to(DEVICE) for data in batch_data]

        end_points = pe_net(batch_data.permute(0, 2, 1).float())

        loss_list = MODEL.get_loss(batch_center, batch_angle_cls, batch_angle_res, batch_size_res, end_points, DEVICE)
        loss, center_loss, stage1_center_loss, h_cls_loss, \
        h_res_loss, s_res_loss, corners_loss = loss_list
        total_loss = loss
        # total_loss = torch.sum(torch.as_tensor(loss_list))
        # total_loss.requires_grad = True

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iou2ds, iou3ds = compute_box3d_iou(
            end_points['center'].detach().cpu().numpy(),
            end_points['angle_scores'].detach().cpu().numpy(), end_points['angle_res'].detach().cpu().numpy(),
            end_points['size_res'].detach().cpu().numpy(),
            batch_center.detach().cpu().numpy(), batch_angle_cls.detach().cpu().numpy(),
            batch_angle_res.detach().cpu().numpy(), batch_size_res.detach().cpu().numpy())
        
        del batch_data
        del batch_center
        del batch_angle_cls
        del batch_angle_res
        del batch_size_res

        end_points['iou2ds'] = iou2ds
        end_points['iou3ds'] = iou3ds

        total_loss_sum += total_loss
        center_loss_sum += center_loss
        stage1_center_loss_sum += stage1_center_loss
        h_cls_loss_sum += h_cls_loss
        h_res_loss_sum += h_res_loss
        s_res_loss_sum += s_res_loss
        corners_loss_sum += corners_loss

        iou2ds_sum += np.sum(iou2ds)
        iou3ds_sum += np.sum(iou3ds)
        iou3d_correct_cnt_50 += np.sum(iou3ds >= 0.5)
        iou3d_correct_cnt_70 += np.sum(iou3ds >= 0.7)

        internal = 50
        if (batch_idx + 1) % internal == 0:
            log_string(' -- %03d / %03d --' % (batch_idx + 1, num_batches))
            log_string('mean total loss: %f' % (total_loss_sum / internal))
            log_string('mean center loss: %f' % (center_loss_sum / internal))
            log_string('mean stage1 center loss: %f' % (stage1_center_loss_sum / internal))
            log_string('mean angle class loss: %f' % (h_cls_loss_sum / internal))
            log_string('mean angle res loss: %f' % (h_res_loss_sum / internal))
            log_string('mean size res loss: %f' % (s_res_loss_sum / internal))
            log_string('mean corners loss: %f' % (corners_loss_sum / internal))

            log_string('box IoU (ground/3D): %f / %f' % \
                       (iou2ds_sum / float(BATCH_SIZE * internal), iou3ds_sum / float(BATCH_SIZE * internal)))
            log_string('box estimation accuracy (IoU=0.5): %f' % \
                       (float(iou3d_correct_cnt_50) / float(BATCH_SIZE * internal)))
            log_string('box estimation accuracy (IoU=0.7): %f' % \
                       (float(iou3d_correct_cnt_70) / float(BATCH_SIZE * internal)))

            total_loss_sum = 0
            center_loss_sum = 0
            stage1_center_loss_sum = 0
            h_cls_loss_sum = 0
            h_res_loss_sum = 0
            s_res_loss_sum = 0
            corners_loss_sum = 0

            iou2ds_sum = 0
            iou3ds_sum = 0
            iou3d_correct_cnt_50 = 0
            iou3d_correct_cnt_70 = 0


def eval_one_epoch(pe_net):
    global EPOCH_CNT
    is_training = False
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))
    val_idxs = np.arange(0, len(VAL_DATASET))
    num_batches = int(np.ceil(len(VAL_DATASET) / BATCH_SIZE_EVAL))

    # To collect statistics

    total_loss_sum = 0
    center_loss_sum = 0
    stage1_center_loss_sum = 0
    h_cls_loss_sum = 0
    h_res_loss_sum = 0
    s_res_loss_sum = 0
    corners_loss_sum = 0

    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt_50 = 0
    iou3d_correct_cnt_70 = 0

    # Simple evaluation with batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE_EVAL
        end_idx = (batch_idx + 1) * BATCH_SIZE_EVAL

        with torch.no_grad():
            batch_data = get_batch(VAL_DATASET, val_idxs, start_idx, end_idx, NUM_CHANNEL)

            batch_data, batch_center, \
            batch_angle_cls, batch_angle_res, batch_size_res = [torch.from_numpy(data).to(DEVICE) for data in batch_data]

            # pe_net.to(torch.device('cpu'))
            end_points = pe_net(batch_data.permute(0, 2, 1).float()) # 

            for key in end_points.keys():
                end_points[key] = end_points[key]#.cpu()

            loss_list = MODEL.get_loss(batch_center, batch_angle_cls, batch_angle_res, batch_size_res, end_points, DEVICE) # torch.device('cpu')
            loss, center_loss, stage1_center_loss, h_cls_loss, \
            h_res_loss, s_res_loss, corners_loss = loss_list
            total_loss = loss

        # val_writer.add_summary(summary, step)

        total_loss_sum += total_loss
        center_loss_sum += center_loss
        stage1_center_loss_sum += stage1_center_loss
        h_cls_loss_sum += h_cls_loss
        h_res_loss_sum += h_res_loss
        s_res_loss_sum += s_res_loss
        corners_loss_sum += corners_loss

        iou2ds, iou3ds = compute_box3d_iou(
            end_points['center'].detach().cpu().numpy(),
            end_points['angle_scores'].detach().cpu().numpy(), end_points['angle_res'].detach().cpu().numpy(),
            end_points['size_res'].detach().cpu().numpy(),
            batch_center.detach().cpu().numpy(), batch_angle_cls.detach().cpu().numpy(),
            batch_angle_res.detach().cpu().numpy(), batch_size_res.detach().cpu().numpy())

        del batch_data
        del batch_center
        del batch_angle_cls
        del batch_angle_res
        del batch_size_res

        iou2ds_sum += np.sum(iou2ds)
        iou3ds_sum += np.sum(iou3ds)
        iou3d_correct_cnt_50 += np.sum(iou3ds >= 0.5)
        iou3d_correct_cnt_70 += np.sum(iou3ds >= 0.7)

    log_string('eval mean total loss: %f' % (total_loss_sum / float(num_batches)))
    log_string('eval mean center loss: %f' % (center_loss_sum / float(num_batches)))
    log_string('eval mean stage1 center loss: %f' % (stage1_center_loss_sum / float(num_batches)))
    log_string('eval mean angle class loss: %f' % (h_cls_loss_sum / float(num_batches)))
    log_string('eval mean angle res loss: %f' % (h_res_loss_sum / float(num_batches)))
    log_string('eval mean size res loss: %f' % (s_res_loss_sum / float(num_batches)))
    log_string('eval mean corners loss: %f' % (corners_loss_sum / float(num_batches)))

    log_string('eval box IoU (ground/3D): %f / %f' % \
               (iou2ds_sum / float(num_batches * BATCH_SIZE_EVAL), iou3ds_sum / \
                float(num_batches * BATCH_SIZE_EVAL)))
    log_string('eval box estimation accuracy (IoU=0.5): %f' % \
               (float(iou3d_correct_cnt_50) / float(num_batches * BATCH_SIZE_EVAL)))
    log_string('eval box estimation accuracy (IoU=0.7): %f' % \
               (float(iou3d_correct_cnt_70) / float(num_batches * BATCH_SIZE_EVAL)))

    EPOCH_CNT += 1


def vis_point_cloud(pcd_numpy):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_numpy[:, :-1])  # pcd_numpy = batch_data[14]
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    train()
    LOG_FOUT.close()