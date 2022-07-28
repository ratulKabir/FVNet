import sys
import os
# import tensorflow as tf
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'kitti'))

import torch_util
from kitti_util import NUM_ANGLE_BIN, g_mean_size_arr
from model_util import parse_output_to_tensors, get_box3d_corners, get_box3d_corners_helper

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Center_Regression_Net(torch.nn.Module):
    """ Regression network for center delta. a.k.a. T-Net.
    Input:
        object_point_cloud: TF tensor in shape (B,M,C)
            point clouds in 3D mask coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
    Output:
        predicted_center: TF tensor in shape (B,3)
    """

    def __init__(self, num_point):
        super().__init__()
        self.conv1 = torch_util.CONV2D(4, 128, 1, padding='valid')
        self.conv2 = torch_util.CONV2D(128, 128, 1, padding='valid')
        self.conv3 = torch_util.CONV2D(128, 256, 1, padding='valid')
        self.conv4 = torch_util.CONV2D(256, 512, 1, padding='valid')
        self.maxpool = torch.nn.MaxPool2d((num_point, 1))
        self.fc0 = torch_util.FCLayer(512, 512)
        self.fc1 = torch_util.FCLayer(512, 256)
        self.fc2 = torch_util.FCLayer(256, 128)
        self.fc3 = torch_util.FCLayer(128, 3)

    def forward(self, x):
        x = torch.unsqueeze(x, 3)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = torch.squeeze(torch.squeeze(x, 2), 2)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class Box_Estimation_Net_3D(torch.nn.Module):
    """ Regression network for center delta. a.k.a. T-Net.
    Input:
        object_point_cloud: TF tensor in shape (B,M,C)
            point clouds in 3D mask coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
    Output:
        predicted_center: TF tensor in shape (B,3)
    """

    def __init__(self, num_point):
        super().__init__()
        self.conv1 = torch_util.CONV2D(4, 128, 1, padding='valid')
        self.conv2 = torch_util.CONV2D(128, 256, 1, padding='valid')
        self.conv3 = torch_util.CONV2D(256, 512, 1, padding='valid')
        self.conv4 = torch_util.CONV2D(512, 1024, 1, padding='valid')
        self.maxpool = torch.nn.MaxPool2d((num_point, 1))
        self.fc0 = torch_util.FCLayer(1024, 1024)
        self.fc1 = torch_util.FCLayer(1024, 512)
        self.fc2 = torch_util.FCLayer(512, 256)
        self.fc3 = torch_util.FCLayer(256, 128)
        self.fc4 = torch_util.FCLayer(128, 3 + NUM_ANGLE_BIN * 2 + 3)

    def forward(self, x):
        x = torch.unsqueeze(x, 3)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = torch.squeeze(torch.squeeze(x, 2), 2)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class Front_PointNet(torch.nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.num_points = num_points
        self.cr_net = Center_Regression_Net(num_points)
        self.be_net = Box_Estimation_Net_3D(num_points)

    def forward(self, point_cloud):
        end_points = {}
        # T-Net and coordinate translation
        center_delta = self.cr_net(point_cloud)
        end_points['center_delta'] = center_delta

        # Get object point cloud in object coordinate
        point_cloud_xyz = point_cloud[:, :, :3]
        point_cloud_features = point_cloud[:, :, 3:]
        point_cloud_xyz_new = point_cloud_xyz - torch.unsqueeze(center_delta, 1)
        point_cloud_new = torch.cat((point_cloud_xyz_new, point_cloud_features), -1)

        # Amodel Box Estimation PointNet
        output = self.be_net(point_cloud_new)

        # Parse output to 3D box parameters
        end_points = parse_output_to_tensors(output, end_points)
        end_points['center'] = end_points['center_res'] + end_points['center_delta']

        return end_points

def get_model(point_cloud):
    num_points = point_cloud.shape[-1]
    end_points = {}
    cr_net = Center_Regression_Net(num_points)
    be_net = Box_Estimation_Net_3D(num_points)

    # T-Net and coordinate translation
    center_delta = cr_net(point_cloud)
    end_points['center_delta'] = center_delta

    # Get object point cloud in object coordinate
    point_cloud_xyz = point_cloud[:, :, :3]
    point_cloud_features = point_cloud[:, :, 3:]
    point_cloud_xyz_new = point_cloud_xyz - torch.unsqueeze(center_delta, 1)
    point_cloud_new = torch.cat((point_cloud_xyz_new, point_cloud_features), -1)

    # Amodel Box Estimation PointNet
    output = be_net(point_cloud_new)

    # Parse output to 3D box parameters
    end_points = parse_output_to_tensors(output, end_points)
    end_points['center'] = end_points['center_res'] + end_points['center_delta']

    return end_points


def huber_loss(error, delta):
    abs_error = torch.abs(error)
    quadratic = torch.minimum(abs_error, torch.tensor(delta))
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + delta * linear
    return torch.mean(losses)


def get_loss(center_label, angle_cls_label, angle_res_label, size_res_label, end_points, device):
    # Center regression losses

    x_dist = torch.linalg.norm(center_label[..., 0] - end_points['center'][..., 0], dim=-1)
    x_loss = huber_loss(x_dist, delta=1.0)
    y_dist = torch.linalg.norm(center_label[..., 1] - end_points['center'][..., 1], dim=-1)
    y_loss = huber_loss(y_dist, delta=1.0)
    z_dist = torch.linalg.norm(center_label[..., 2] - end_points['center'][..., 2], dim=-1)
    z_loss = huber_loss(z_dist, delta=1.0)
    center_loss = x_loss + y_loss + z_loss
    # center_dist = tf.norm(center_label - end_points['center'], axis=-1)
    # center_loss = huber_loss(center_dist, delta=2.0)
    # tf.summary.scalar('center_loss', center_loss)                         # goes  into tensorboard

    stage1_x_dist = torch.linalg.norm(center_label[..., 0] - end_points['center_delta'][..., 0], dim=-1)
    stage1_x_loss = huber_loss(stage1_x_dist, delta=1.0)
    stage1_y_dist = torch.linalg.norm(center_label[..., 1] - end_points['center_delta'][..., 1], dim=-1)
    stage1_y_loss = huber_loss(stage1_y_dist, delta=1.0)
    stage1_z_dist = torch.linalg.norm(center_label[..., 2] - end_points['center_delta'][..., 2], dim=-1)
    stage1_z_loss = huber_loss(stage1_z_dist, delta=1.0)

    # stage1_center_dist = tf.norm(center_label - end_points['center_delta'], axis=-1)
    # stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)
    stage1_center_loss = stage1_x_loss + stage1_y_loss + stage1_z_loss
    # tf.summary.scalar('stage1 center loss', stage1_center_loss)   # goes  into tensorboard

    # Heading angle loss
    angle_cls_loss = mean_sparse_softmax_cross_entropy_with_logits(
        logits=end_points['angle_scores'], labels=angle_cls_label.type(torch.LongTensor).to(device))
    # tf.summary.scalar('angle_class_loss', angle_cls_loss)   # goes  into tensorboard

    hcls_onehot = torch.nn.functional.one_hot(angle_cls_label.type(torch.LongTensor),
                                              num_classes=NUM_ANGLE_BIN).to(device)  # BxNUM_ANGLE_BIN
    angle_per_class = 2 * np.pi / NUM_ANGLE_BIN
    angle_res_normalized_label = angle_res_label / (angle_per_class / 2)
    # print(end_points['angle_res_normalized'].is_cuda)
    # print(hcls_onehot.is_cuda)
    angle_res_normalized_loss = huber_loss(
        torch.mean(end_points['angle_res_normalized'] * hcls_onehot.float(), dim=1) - \
        angle_res_normalized_label, delta=1.0)
    # tf.summary.scalar('angle_res_loss', angle_res_normalized_loss)

    # Size loss
    mean_sizes = torch.unsqueeze(torch.tensor(g_mean_size_arr, dtype=torch.float32, device=device), 0)  # (1,NS,3)

    size_res_label_normalized = size_res_label / mean_sizes
    # size_normalized_dist = tf.norm( \
    #     size_res_label_normalized - end_points['size_res_normalized'],
    #     axis=-1)
    l_dist = torch.norm(size_res_label_normalized[..., 0] - end_points['size_res_normalized'][..., 0], dim=-1)
    l_loss = huber_loss(l_dist, delta=1.0)
    w_dist = torch.norm(size_res_label_normalized[..., 1] - end_points['size_res_normalized'][..., 1], dim=-1)
    w_loss = huber_loss(w_dist, delta=1.0)
    h_dist = torch.norm(size_res_label_normalized[..., 2] - end_points['size_res_normalized'][..., 2], dim=-1)
    h_loss = huber_loss(h_dist, delta=1.0)
    # size_res_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)
    size_res_normalized_loss = l_loss + w_loss + h_loss
    # tf.summary.scalar('size_res_loss', size_res_normalized_loss)

    # Corner loss
    corners_3d = get_box3d_corners(end_points['center'], end_points['angle_res'], end_points['size_res'], device)  # (B,NH,8,3)

    corners_3d_pred = torch.mean(torch.unsqueeze(torch.unsqueeze(hcls_onehot, -1), -1).float() * corners_3d,
                                 dim=[1])  # (B,8,3)

    angle_bin_centers = torch.tensor(np.arange(0, 2 * np.pi, 2 * np.pi / NUM_ANGLE_BIN),
                                     dtype=torch.float32, device=device)  # (NH,)
    heading_label = torch.unsqueeze(angle_res_label, 1) + \
                    torch.unsqueeze(angle_bin_centers, 0)  # (B,NH)
    heading_label = torch.mean(hcls_onehot.float() * heading_label, dim=1)

    size_label = mean_sizes + size_res_label

    corners_3d_gt = get_box3d_corners_helper(center_label, heading_label, size_label, device)  # (B,8,3)
    corners_3d_gt_flip = get_box3d_corners_helper(center_label, heading_label + np.pi, size_label, device)  # (B,8,3)

    corners_dist = torch.minimum(torch.linalg.norm(corners_3d_pred - corners_3d_gt, dim=-1),
                                 torch.linalg.norm(corners_3d_pred - corners_3d_gt_flip, dim=-1))
    # corners_dist = tf.norm(corners_3d_pred - corners_3d_gt, axis=-1)
    corners_loss = huber_loss(corners_dist, delta=1.0)
    # tf.summary.scalar('corners_loss', corners_loss)

    total_loss = center_loss + \
                 stage1_center_loss + \
                 angle_cls_loss + \
                 20.0 * angle_res_normalized_loss + \
                 20.0 * size_res_normalized_loss + \
                 5.0 * corners_loss

    # tf.add_to_collection('losses', total_loss)

    return [total_loss, center_loss, stage1_center_loss, angle_cls_loss,
            angle_res_normalized_loss, size_res_normalized_loss, corners_loss]


def mean_sparse_softmax_cross_entropy_with_logits(logits, labels):
    log_sm = torch.nn.LogSoftmax(dim=1)
    loss_fn = torch.nn.NLLLoss()

    return loss_fn(log_sm(logits), labels)
