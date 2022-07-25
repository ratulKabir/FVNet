import os
import sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'kitti'))
from kitti_util import NUM_ANGLE_BIN, g_mean_size_arr

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_OBJECT_POINT = 512


def parse_output_to_tensors(output, end_points):
    batch_size = output.shape[0]
    center_res = output[:, :3]

    end_points['center_res'] = center_res

    angle_scores = output[:, 3:3+NUM_ANGLE_BIN]
    angle_res_norm = output[:, 3+NUM_ANGLE_BIN:3+NUM_ANGLE_BIN+NUM_ANGLE_BIN]
    end_points['angle_scores'] = angle_scores  # BxNUM_ANGLE_BIN
    end_points['angle_res_normalized'] = angle_res_norm  # BxNUM_ANGLE_BIN (-1 to 1)
    angle_per_class = 2 * np.pi / NUM_ANGLE_BIN
    end_points['angle_res'] = angle_res_norm * (angle_per_class / 2)  # BxNUM_ANGLE_BIN

    size_res_norm = output[:, 3+NUM_ANGLE_BIN*2:(3+NUM_ANGLE_BIN*2)+3]
    size_res_norm = torch.reshape(size_res_norm, (batch_size, 3))  # Bx3
    end_points['size_res_normalized'] = size_res_norm
    end_points['size_res'] = size_res_norm * torch.unsqueeze(torch.tensor(g_mean_size_arr, dtype=torch.float32, device=DEVICE), 0)

    return end_points


def get_box3d_corners(center, angle_res, size_res, device):
    batch_size = center.shape[0]
    angle_bin_centers = torch.tensor(np.arange(0, 2*np.pi, 2 * np.pi / NUM_ANGLE_BIN), dtype=torch.float32, device=device)  # (NH,)
    angles = angle_res + torch.unsqueeze(angle_bin_centers, 0)  # (B,NH)

    mean_sizes = torch.unsqueeze(torch.tensor(g_mean_size_arr, dtype=torch.float32, device=device), 0)  # (1,3)
    sizes = mean_sizes + size_res  # (B,3)
    sizes = torch.tile(torch.unsqueeze(sizes, 1), (1, NUM_ANGLE_BIN, 1))  # (B,NH,3)

    centers = torch.tile(torch.unsqueeze(center, 1), (1, NUM_ANGLE_BIN, 1))  # (B,NH,3)

    N = batch_size * NUM_ANGLE_BIN
    corners_3d = get_box3d_corners_helper(torch.reshape(centers, (N, 3)), torch.reshape(angles, (N,)),
                                          torch.reshape(sizes, (N, 3)), device)

    return torch.reshape(corners_3d, (batch_size, NUM_ANGLE_BIN, 8, 3))


def get_box3d_corners_helper(centers, angles, sizes, device):
    N = centers.shape[0]
    l = sizes[:, :1]
    w = sizes[:, 1:2]
    h = sizes[:, 2:3]
    # print l,w,h
    x_corners = torch.cat((l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2), dim=1)  # (N,8)
    y_corners = torch.cat((h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2), dim=1)  # (N,8)
    z_corners = torch.cat((w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2), dim=1)  # (N,8)
    corners = torch.cat((torch.unsqueeze(x_corners, 1), torch.unsqueeze(y_corners, 1), torch.unsqueeze(z_corners, 1)),
                        dim=1)  # (N,3,8)
    # print x_corners, y_corners, z_corners
    c = torch.cos(angles)
    s = torch.sin(angles)
    ones = torch.ones([N], dtype=torch.float32, device=device)
    zeros = torch.zeros([N], dtype=torch.float32, device=device)
    row1 = torch.stack((c, zeros, s), dim=1)  # (N,3)
    row2 = torch.stack((zeros, ones, zeros), dim=1)
    row3 = torch.stack((-s, zeros, c), dim=1)
    R = torch.cat((torch.unsqueeze(row1, 1), torch.unsqueeze(row2, 1), torch.unsqueeze(row3, 1)), dim=1)  # (N,3,3)
    # print row1, row2, row3, R, N
    corners_3d = torch.matmul(R, corners)  # (N,3,8)
    corners_3d += torch.tile(torch.unsqueeze(centers, 2), (1, 1, 8))  # (N,3,8)
    corners_3d = corners_3d.permute(0, 2, 1) # should be same as transpose
    return corners_3d





