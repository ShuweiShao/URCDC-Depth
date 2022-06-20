import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms

import numpy as np
from PIL import Image
import os
import random

from utils import DistributedSamplerNoEvenlyDivisible
import copy

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class NewDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None
    
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                # self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)
        
        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))
            
            
class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()
    
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
        self.count = 0
        self.sample_nums = len(self.filenames)
    
    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        sample_path_random = self.filenames[random.randint(0,len(self.filenames)-1)]
        # focal = float(sample_path.split()[2])
        focal = 518.8579

        if self.mode == 'train':
            if self.args.dataset == 'kitti':
                rgb_file = sample_path.split()[0]
                depth_file = os.path.join(sample_path.split()[0].split('/')[0], sample_path.split()[1])
                if self.args.use_right is True and random.random() > 0.5:
                    rgb_file.replace('image_02', 'image_03')
                    depth_file.replace('image_02', 'image_03')
                # new add
                rgb_file_random = sample_path_random.split()[0]
                depth_file_random = os.path.join(sample_path_random.split()[0].split('/')[0], sample_path_random.split()[1])
                if self.args.use_right is True and random.random() > 0.5:
                    rgb_file_random.replace('image_02', 'image_03')
                    depth_file_random.replace('image_02', 'image_03')            

            else:
                rgb_file = sample_path.split()[0]
                depth_file = sample_path.split()[1]
                # new add
                rgb_file_random = sample_path_random.split()[0]
                depth_file_random = sample_path_random.split()[1]


            image_path = os.path.join(self.args.data_path, rgb_file)
            depth_path = os.path.join(self.args.gt_path, depth_file)
            # new add
            image_path_random = os.path.join(self.args.data_path, rgb_file_random)
            depth_path_random = os.path.join(self.args.gt_path, depth_file_random)            
            
            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)
            # new add
            image_random = Image.open(image_path_random)
            depth_gt_random = Image.open(depth_path_random)
            
            if self.args.do_kb_crop is True:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                # new add
                height_random = image_random.height
                width_random = image_random.width
                top_margin_random = int(height_random - 352)
                left_margin_random = int((width_random - 1216) / 2)
                depth_gt_random = depth_gt_random.crop((left_margin_random, top_margin_random, left_margin_random + 1216, top_margin_random + 352))
                image_random = image_random.crop((left_margin_random, top_margin_random, left_margin_random + 1216, top_margin_random + 352))
            
            # To avoid blank boundaries due to pixel registration
            if self.args.dataset == 'nyu':
                if self.args.input_height == 480:
                    depth_gt = np.array(depth_gt)
                    valid_mask = np.zeros_like(depth_gt)
                    valid_mask[45:472, 43:608] = 1
                    depth_gt[valid_mask==0] = 0
                    depth_gt = Image.fromarray(depth_gt)
                    # new add
                    depth_gt_random = np.array(depth_gt_random)
                    valid_mask_random = np.zeros_like(depth_gt_random)
                    valid_mask_random[45:472, 43:608] = 1
                    depth_gt_random[valid_mask_random==0] = 0
                    depth_gt_random = Image.fromarray(depth_gt_random)                    
                else:
                    depth_gt = depth_gt.crop((43, 45, 608, 472))
                    image = image.crop((43, 45, 608, 472))
                    # new add
                    depth_gt_random = depth_gt_random.crop((43, 45, 608, 472))
                    image_random = image_random.crop((43, 45, 608, 472))
    
            if self.args.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
                # new add
                image_random = self.rotate_image(image_random, random_angle)
                depth_gt_random = self.rotate_image(depth_gt_random, random_angle, flag=Image.NEAREST)                
            
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            # new add
            image_random = np.asarray(image_random, dtype=np.float32) / 255.0
            depth_gt_random = np.asarray(depth_gt_random, dtype=np.float32)
            depth_gt_random = np.expand_dims(depth_gt_random, axis=2)


            if self.args.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
                # new add
                depth_gt_random = depth_gt_random / 1000.0
            else:
                depth_gt = depth_gt / 256.0
                # new add
                depth_gt_random = depth_gt_random / 256.0

            if image.shape[0] != self.args.input_height or image.shape[1] != self.args.input_width:
                image, depth_gt = self.random_crop(image, depth_gt, self.args.input_height, self.args.input_width)
                # new add
                image_random, depth_gt_random = self.random_crop(image_random, depth_gt_random, self.args.input_height, self.args.input_width)
            image,depth_gt = self.graft(image,depth_gt,image_random, depth_gt_random)
            image,depth_gt = self.split_flip(image,depth_gt)
            image,image2, depth_gt = self.train_preprocess(image, depth_gt)
            sample = {'image': image, 'image2': image2,'depth': depth_gt, 'focal': focal}
        
        else:
            if self.mode == 'online_eval':
                data_path = self.args.data_path_eval
            else:
                data_path = self.args.data_path

            image_path = os.path.join(data_path, "./" + sample_path.split()[0])
            image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

            if self.mode == 'online_eval':
                gt_path = self.args.gt_path_eval
                depth_path = os.path.join(gt_path, "./" + sample_path.split()[1])
                if self.args.dataset == 'kitti':
                    depth_path = os.path.join(gt_path, sample_path.split()[0].split('/')[0], sample_path.split()[1])
                has_valid_depth = False
                try:
                    depth_gt = Image.open(depth_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    if self.args.dataset == 'nyu':
                        depth_gt = depth_gt / 1000.0
                    else:
                        depth_gt = depth_gt / 256.0

            if self.args.do_kb_crop is True:
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
            
            if self.mode == 'online_eval':
                sample = {'image': image,'image2': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth}
            else:
                sample = {'image': image, 'focal': focal}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
        
        # image = self.cutdepth(image, depth_gt)
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        # do_augment2 = random.random()
        
        if do_augment > 0.5:
            image = self.augment_image(image)
        image2 = copy.deepcopy(image)          
        # if do_augment2 > 0.5:
        #     image2 = self.augment_image(image2)

        return image,image2,depth_gt
    
    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
    
    def split_flip(self,image,depth):

        p = random.random()
        if p<0.5:
            return image,depth
        image_copy = copy.deepcopy(image)
        depth_copy = copy.deepcopy(depth)
        h,w,c = image.shape
        N = 2     # split numbers
        h_list=[]      
        h_interval_list = []        # hight interval
        if N==2:
            h_list.append(random.randint(int(0.2*h), int(0.8*h)))
        elif N==3:
            h_list.append(random.randint(int(0.2*h), int(0.4*h)))
            h_list.append(random.randint(int(0.6*h), int(0.8*h)))
        elif N==4:
            h_list.append((int(0.25*h)))
            h_list.append((int(0.5*h)))
            h_list.append((int(0.75*h)))        
        h_list.append(h)
        h_list.append(0)  
        h_list.sort()
        h_list_inv = np.array([h]*(N+1))-np.array(h_list)
        for i in range(len(h_list)-1):
            h_interval_list.append(h_list[i+1]-h_list[i])

        for i in range(N):
            image[h_list[i]:h_list[i+1],:,:] = image_copy[h_list_inv[i]-h_interval_list[i]:h_list_inv[i],:,:]
            depth[h_list[i]:h_list[i+1],:,:] = depth_copy[h_list_inv[i]-h_interval_list[i]:h_list_inv[i],:,:]
        return image,depth

    def graft(self,image,depth,image_random,depth_random):
        if random.random()<0.5:
            return image,depth
        h,w,c = image.shape
        h_graft = random.randint(int(0.4*h),int(0.6*h))
        p = random.random()
        if p<0.25:
            image[:h_graft,:,:] = image_random[:h_graft,:,:]
            depth[:h_graft,:,:] = depth_random[:h_graft,:,:]
        elif p>0.25 and p<0.5:
            image[:h_graft,:,:] = image_random[h-h_graft:,:,:]
            depth[:h_graft,:,:] = depth_random[h-h_graft:,:,:]
        elif p>0.5 and p<0.75:
            image[h_graft:,:,:] = image_random[h_graft:,:,:]
            depth[h_graft:,:,:] = depth_random[h_graft:,:,:]
        else:
            image[h_graft:,:,:] = image_random[:h-h_graft,:,:]
            depth[h_graft:,:,:] = depth_random[:h-h_graft,:,:]

        return image,depth

    def __len__(self):
        return len(self.filenames)

    def cutdepth(self, image, depth):
        H, W, C = image.shape

        if self.count % 4 == 0:
            alpha = random.random()
            beta = random.random()
            p = 0.75

            l = int(alpha * W)
            w = int(max((W - alpha * W) * beta * p, 1))
            if self.args.dataset == 'nyu':
                image[:, l:l+w, 0] = depth[:, l:l+w, 0] / 10.0
                image[:, l:l+w, 1] = depth[:, l:l+w, 0] / 10.0
                image[:, l:l+w, 2] = depth[:, l:l+w, 0] / 10.0
            else :
                image[:, l:l+w, 0] = depth[:, l:l+w, 0] / 80.0
                image[:, l:l+w, 1] = depth[:, l:l+w, 0] / 80.0
                image[:, l:l+w, 2] = depth[:, l:l+w, 0] / 80.0
        self.count += 1

        return image

class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, sample):
        image, image2,focal = sample['image'], sample['image2'],sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)
        image2 = self.to_tensor(image2)
        image2 = self.normalize(image2)


        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'image2': image2,'depth': depth, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth}
    
    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img
        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
