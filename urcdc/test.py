from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import os, sys, errno
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from utils import post_process_depth, flip_lr
from networks.NewCRFDepth import NewCRFDepth


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='URCDC PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help='model name', default='urcdc')
parser.add_argument('--encoder', type=str, help='type of encoder, base07, large07', default='large07')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--dataset', type=str, help='dataset to train on', default='nyu')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--save_viz', help='if set, save visulization of the outputs', action='store_true')

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.dataset == 'kitti' or args.dataset == 'nyu' or args.dataset == 'kitti_benchmark' or args.dataset == 'nyu_beihang' or args.dataset == 'kitti_beihang':
    from dataloaders.dataloader import NewDataLoader
elif args.dataset == 'kittipred':
    from dataloaders.dataloader_kittipred import NewDataLoader

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def test(params):
    """Test function."""
    args.mode = 'test'
    dataloader = NewDataLoader(args, 'test')
    
    model = NewCRFDepth(version='large07', inv_depth=False, max_depth=args.max_depth)
    model = torch.nn.DataParallel(model)
    
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_test_samples = get_num_lines(args.filenames_file)

    with open(args.filenames_file) as f:
        lines = f.readlines()

    print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))

    pred_depths = []
    u1_list = []
    u2_list = []
    start_time = time.time()
    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader.data)):
            image = Variable(sample['image'].cuda())
            # Predict
            preds = model(image)
            pred_depth = preds['pred_d']
            u_1 = preds['u1'] 
            u_2 = preds['u2'] 

            post_process = True
            if post_process:
                image_flipped = flip_lr(image)
                preds2 = model(image_flipped)
                pred_depth_flipped = preds2['pred_d']
                depth_est = post_process_depth(pred_depth, pred_depth_flipped)

            pred_depth = depth_est.cpu().numpy().squeeze()
            u_1 = u_1.cpu().numpy().squeeze()
            u_2 = u_2.cpu().numpy().squeeze()

            if args.do_kb_crop:
                height, width = 352, 1216
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
                pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
                pred_depth = pred_depth_uncropped

            pred_depths.append(pred_depth)
            u1_list.append(u_1)
            u2_list.append(u_2)

    elapsed_time = time.time() - start_time
    print('Elapesed time: %s' % str(elapsed_time))
    print('Done.')
    exp_name = '%s'%(datetime.now().strftime('%m%d'))
    save_name = 'models/%s/result_'%(exp_name) + args.model_name
    
    print('Saving result pngs..')
    if not os.path.exists(save_name):
        try:
            os.mkdir(save_name)
            os.mkdir(save_name + '/raw')
            os.mkdir(save_name + '/cmap')
            os.mkdir(save_name + '/rgb')
            os.mkdir(save_name + '/gt')
            os.mkdir(save_name + '/u1')
            os.mkdir(save_name+'/u2')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    
    for s in tqdm(range(num_test_samples)):
        if args.dataset == 'kitti' or args.dataset == 'kitti_beihang':
            date_drive = lines[s].split('/')[1]
            filename_pred_png = save_name + '/raw/' + date_drive + '_' + lines[s].split()[0].split('/')[-1].replace(
                '.png', '_urcd.png')
            filename_cmap_png = save_name + '/cmap/' + date_drive + '_' + lines[s].split()[0].split('/')[
                -1].replace('.png', '_urcd.png')
            filename_image_png = save_name + '/rgb/' + date_drive + '_' + lines[s].split()[0].split('/')[-1]
            filename_u1 = save_name + '/u1/' + date_drive + '_' + lines[s].split()[0].split('/')[-1].replace(
                '.png', '_u1.png')
            filename_u2 = save_name + '/u2/' + date_drive + '_' + lines[s].split()[0].split('/')[-1].replace(
                '.png', '_u2.png')    
        elif args.dataset == 'kitti_benchmark':
            filename_pred_png = save_name + '/raw/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
            filename_cmap_png = save_name + '/cmap/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
            filename_image_png = save_name + '/rgb/' + lines[s].split()[0].split('/')[-1]
            filename_u1 = save_name + '/u1/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
            filename_u2 = save_name + '/u2/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
        elif args.dataset == 'nyu_beihang':
            scene_name = lines[s].split()[0].split('/')[0]
            filename_pred_png = save_name + '/raw/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace('.jpg', '.png')
            filename_cmap_png = save_name + '/cmap/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace('.jpg', '_urcd.png')            
        else:
            scene_name = lines[s].split()[0].split('/')[0]
            filename_pred_png = save_name + '/raw/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
                '.jpg', '.png')
            filename_cmap_png = save_name + '/cmap/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
                '.jpg', '_urcd.png')
            filename_gt_png = save_name + '/gt/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
                '.jpg', '_gt.png')
            filename_image_png = save_name + '/rgb/' + scene_name + '_' + lines[s].split()[0].split('/rgb_')[1]
            filename_u1 = save_name + '/u1/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
                '.jpg', '_u1.png')
            filename_u2 = save_name + '/u2/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
                '.jpg', '_u2.png')
        rgb_path = os.path.join(args.data_path, './' + lines[s].split()[0])
        image = cv2.imread(rgb_path)
        if args.dataset == 'nyu' or args.dataset == 'nyu_beihang':
            gt_path = os.path.join(args.data_path, './' + lines[s].split()[1])
            gt = cv2.imread(gt_path, -1).astype(np.float32) / 1000.0  # Visualization purpose only
            # gt[gt == 0] = np.amax(gt)
        
        pred_depth = pred_depths[s]
        u_1 = u1_list[s]
        u_2 = u2_list[s]

        
        if args.dataset == 'kitti' or args.dataset == 'kitti_benchmark' or args.dataset == 'kitti_beihang':
            pred_depth_scaled = pred_depth * 256.0
        else:
            pred_depth_scaled = pred_depth * 1000.0

        
        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
        cv2.imwrite(filename_pred_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        if args.save_viz:
            # cv2.imwrite(filename_image_png, image[10:-1 - 9, 10:-1 - 9, :])
            if args.dataset == 'nyu':
                gt = np.where(gt<1e-3,gt*0+1e-3,gt)
                plt.imsave(filename_gt_png, gt, cmap='jet')
                pred_depth_cropped = pred_depth[10:-1 - 9, 10:-1 - 9]
                plt.imsave(filename_cmap_png, np.log10(pred_depth[10:-1 - 9, 10:-1 - 9]), cmap='jet')
                plt.imsave(filename_u1, u_1[10:-1 - 9, 10:-1 - 9], cmap='jet')
                plt.imsave(filename_u2, u_2[10:-1 - 9, 10:-1 - 9], cmap='jet')
            elif args.dataset == 'nyu_beihang':
                plt.imsave(filename_cmap_png, np.log10(pred_depth), cmap='jet')
                print("max depth:%.2f m"%(pred_depth.max()))
            elif args.dataset == 'kitti_beihang':
                plt.imsave(filename_cmap_png, np.log10(pred_depth), cmap='magma')
                print("max depth:%.2f m"%(pred_depth.max()))
            else:
                plt.imsave(filename_cmap_png, np.log10(pred_depth), cmap='magma')
                # plt.imsave(filename_cmap_png, pred_depth, cmap='magma')
                plt.imsave(filename_u1, u_1, cmap='jet')
                plt.imsave(filename_u2, u_2, cmap='jet')
    
    return


if __name__ == '__main__':
    test(args)
