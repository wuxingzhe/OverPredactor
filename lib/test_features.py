import numpy as np
import os, glob
import random

import sys
sys.path.append('../')
from lib.benchmark_utils import ransac_pose_estimation, random_sample, get_angle_deviation, to_o3d_pcd, to_array

pairs_file = '/home2/wuxingzhe/KITTI/dataset/pairs.txt'
pcd_feature_path = '/home2/wuxingzhe/KITTI/results_kpconv'
pairs = open(pairs_file).readlines()
pairs = [pair.strip() for pair in pairs]
random.shuffle(pairs)
pairs = pairs[:500]

for pair in pairs:
    [seq_name, l, r] = pair.split()
    if not os.path.exists(os.path.join(pcd_feature_path, seq_name, l+'_'+r+'.npy')):   
        continue                                                                                                                                                
    res = np.load(os.path.join(pcd_feature_path, seq_name, l+'_'+r+'.npy'), allow_pickle=True).item()
    src_pcd = res['src_pcd']; tgt_pcd = res['tgt_pcd']
    src_feats = res['src_feats']; tgt_feats = res['tgt_feats']
    M = res['M']
    rot_gt = M[:3, :3]; trans_gt = M[:3, 3]

    distance_threshold = 0.3
    ts_est = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=False, distance_threshold=distance_threshold, ransac_n = 4)

    rot_threshold = 5; trans_threshold = 2
    r_deviation = get_angle_deviation(ts_est[None,:3,:3], rot_gt[None,:,:])
    translation_errors = np.linalg.norm(ts_est[:3, 3]-trans_gt)
    print(seq_name+' '+l+' '+r+' '+str(r_deviation)+' '+str(translation_errors))
