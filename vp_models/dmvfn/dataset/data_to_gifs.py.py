import cv2
import os
import json
import random
import numpy as np
import torch
import h5py
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from collections.abc import Sequence
from absl import app
import tensorflow_datasets as tfds
import tensorflow as tf
import imageio


from tqdm import tqdm




os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CityTrainDataset(Dataset):
    def __init__(self):
        self.path = './data/cityscapes/train'
        self.train_data = sorted(os.listdir(self.path))


    def __len__(self):
        return len(self.train_data)

    def aug_seq(self, imgs, h, w):
        ih, iw, _ = imgs[0].shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x+h, y:y+w, :]
        return imgs

    def getimg(self, index):
        data_name = self.train_data[index]
        data_path = os.path.join(self.path, data_name)
        frame_list = sorted(os.listdir(data_path))
        imgs = []
        for i in range(30):
            im = cv2.imread(os.path.join(data_path, frame_list[i]))
            imgs.append(im)
        return  imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        imgs = self.aug_seq(imgs, 256, 256)
        length = len(imgs)
        if random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_CLOCKWISE)
        elif random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_180)
        elif random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][:, :, ::-1]
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][::-1]
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][:, ::-1]
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)
        return torch.stack(imgs, 0)#n, c, h , w

class CityValDataset(Dataset):
    def __init__(self):
        self.val_data = []
        self.video_path = './data/cityscapes/test'
        self.video_data = sorted(os.listdir(self.video_path))
        for i in self.video_data:
            self.val_data.append(os.path.join(self.video_path, i))
        self.val_data = sorted(self.val_data)

    def __len__(self):
        return len(self.val_data)
        
    def getimg(self, index):
        data = self.val_data[index]
        img_list = sorted(os.listdir(data))
        imgs = []
        for i in range(14):
            im = cv2.imread(os.path.join(data, img_list[i]))
            imgs.append(im)
        return  imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        name = self.video_data[index]
        length = len(imgs)
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)

        return torch.stack(imgs, 0), name#n, c, h , w

class KittiTrainDataset(Dataset):
    def __init__(self):
        self.path = './data/KITTI/train'
        self.train_data = sorted(os.listdir(self.path))

    def __len__(self):
        return len(self.train_data)


    def aug_seq(self, imgs, h, w):
        ih, iw, _ = imgs[0].shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x+h, y:y+w, :]
        return imgs

    def getimg(self, index):
        data_name = self.train_data[index]
        data_path = os.path.join(self.path, data_name)
        frame_list = sorted(os.listdir(data_path))
        imgs = []
        for i in range(9):
            im = cv2.imread(os.path.join(data_path, frame_list[i]))
            imgs.append(im)
        return  imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        imgs = self.aug_seq(imgs, 256, 256)
        length = len(imgs)
        if random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_CLOCKWISE)
        elif random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_180)
        elif random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][:, :, ::-1]
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][::-1]
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][:, ::-1]
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)

        return torch.stack(imgs, 0)#n, c, h , w

class KittiValDataset(Dataset):
    def __init__(self):
        self.val_data = []
        self.video_path = './data/KITTI/test'
        self.video_data = sorted(os.listdir(self.video_path))
        for i in self.video_data:
            self.val_data.append(os.path.join(self.video_path, i))
        self.val_data = sorted(self.val_data)

    def __len__(self):
        return len(self.val_data)
        
    def getimg(self, index):
        data = self.val_data[index]
        img_list = sorted(os.listdir(data))
        imgs = []
        for i in range(9):
            im = cv2.imread(os.path.join(data, img_list[i]))
            imgs.append(im)
        return  imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        name = self.video_data[index]
        length = len(imgs)
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)

        return torch.stack(imgs, 0), name#n, c, h , w

class VimeoTrainDataset(Dataset):
    def __init__(self):
        self.path = './data/Vimeo/sequences/'
        self.train_data = []
        video_paths = sorted(os.listdir(self.path))
        for i in video_paths:
            video_path_1 = sorted(os.listdir(os.path.join(self.path, i)))
            for j in video_path_1:
                self.train_data.append(os.path.join(os.path.join(self.path, i), j))
        

    def __len__(self):
        return len(self.train_data)         

    def aug(self, img0, img1, gt, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, img1, gt

    def getimg(self, index):
        data = self.train_data[index]
        frame_list = sorted(os.listdir(data))
        ind = [0, 1, 2, 3, 4]
        random.shuffle(ind)
        ind = ind[0]
        img0 = cv2.imdread(frame_list[ind])
        img1 = cv2.imdread(frame_list[ind+1])
        gt = cv2.imdread(frame_list[ind+2])
        return img0, img1, gt
            
    def __getitem__(self, index):
        img0, img1, gt = self.getimg(index)
        img0, img1, gt = self.aug(img0, img1, gt, 224, 224)
        if random.randint(0, 1):
            img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
            gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
            img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
        elif random.randint(0, 1):
            img0 = cv2.rotate(img0, cv2.ROTATE_180)
            gt = cv2.rotate(gt, cv2.ROTATE_180)
            img1 = cv2.rotate(img1, cv2.ROTATE_180)
        elif random.randint(0, 1):
            img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
            gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if random.uniform(0, 1) < 0.5:
            img0 = img0[:, :, ::-1]
            img1 = img1[:, :, ::-1]
            gt = gt[:, :, ::-1]
        if random.uniform(0, 1) < 0.5:
            img0 = img0[::-1]
            img1 = img1[::-1]
            gt = gt[::-1]
        if random.uniform(0, 1) < 0.5:
            img0 = img0[:, ::-1]
            img1 = img1[:, ::-1]
            gt = gt[:, ::-1]
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.stack((img0, img1, gt), 0)


class UCFTrainDataset(Dataset):
    def __init__(self):
        self.path = './data/ucf101_jpeg/jpegs_256/'
        self.train_data = []
        with open(os.path.join('/home/huxiaotao/trainlist01.txt')) as f:
            for line in f:
                video_dir = line.rstrip().split('.')[0]
                video_name = video_dir.split('/')[1]
                self.train_data.append(video_name)

    def __len__(self):
        return len(self.train_data)

    def aug_seq(self, imgs, h, w):
        ih, iw, _ = imgs[0].shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x+h, y:y+w, :]
        return imgs

    def getimg(self, index):
        video_path = self.train_data[index]
        frame_list = sorted(os.listdir(os.path.join(self.path, video_path)))
        n = len(frame_list)
        max_time = 5 if 5 <= n/10 else int(n/10)
        time_step = np.random.randint(1, max_time + 1)#1, 2, 3, 4, 5
        frame_ind = np.random.randint(0, n-9*time_step)
        frame_inds = [frame_ind+j*time_step for j in range(10)]
        imgs = []
        for i in frame_inds:
            im = cv2.imread(os.path.join(os.path.join(self.path, video_path), frame_list[i]))
            imgs.append(im)
        return  imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        imgs = self.aug_seq(imgs, 256, 256)
        length = len(imgs)
        if random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_CLOCKWISE)
        elif random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_180)
        elif random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][:, :, ::-1]
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][::-1]
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][:, ::-1]
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)

        return torch.stack(imgs, 0)#n, c, h , w

class VimeoValDataset(Dataset):
    def __init__(self):
        self.val_data = []
        self.video_path = './data/vimeo_interp_test/target/'
        self.data_name = []
        self.video_list = sorted(os.listdir(self.video_path))
        for i in self.video_list:
            self.video_clip_list = sorted(os.listdir(os.path.join(self.video_path, i)))
            for j in self.video_clip_list:
                self.val_data.append(os.path.join(self.video_path, os.path.join(i, j)))
                self.data_name.append(os.path.join(i, j))
        self.val_data = sorted(self.val_data)
        self.data_name = sorted(self.data_name)

    def __len__(self):
        return len(self.val_data)
        
    def getimg(self, index):
        data = self.val_data[index]
        img_list = sorted(os.listdir(data))
        imgs = []
        for i in range(3):
            im = cv2.imread(os.path.join(data, img_list[i]))
            imgs.append(im)
        return  imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        data_name = self.data_name[index]
        length = len(imgs)
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)
        return torch.stack(imgs, 0), data_name#n, c, h , w

class DavisValDataset(Dataset):
    def __init__(self):
        self.val_data = []
        self.video_path = './data/DAVIS/'
        self.video_data = sorted(os.listdir(self.video_path))
        for i in self.video_data:
            self.val_data.append(os.path.join(self.video_path, i))
        self.val_data = sorted(self.val_data)

    def __len__(self):
        return len(self.val_data)
        
    def getimg(self, index):
        data = self.val_data[index]
        img_list = sorted(os.listdir(data)) #一定要sort
        imgs = []
        for i in range(9):
            im = cv2.imread(os.path.join(data, img_list[i]))
            imgs.append(im)
        return  imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        name = self.video_data[index]
        length = len(imgs)
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)

        return torch.stack(imgs, 0), name#n, c, h , w
    
class RobodeskTrainDataset(Dataset):
    def __init__(self):
        self.path = './data/robodesk/'

        self.seq_length = 9

        self.videos_path = os.path.join(self.path, 'videos')

        self.index_path=os.path.join(self.path, 'robodesk_v2.json')

        with open(self.index_path, 'r') as f:
            self.train_data = json.load(f)

        all_files = np.unique([self.train_data[i]['file'] for i in range(len(self.train_data))])

        self.all_files = {}
        for file in all_files:
            file = os.path.join(self.path, file.split("/robodesk/")[1])
            self.all_files[file] = h5py.File(file, 'r')

    def __len__(self):
        return len(self.train_data)         

    def aug_seq(self, imgs, h, w):
        ih, iw, _ = imgs[0].shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x+h, y:y+w, :]
        return imgs


    def getimg(self, index):
        sample = self.train_data[index]
        demo_id = sample['demo']
        
        
        f = self.all_files[os.path.join(self.path, sample['file'].split("/robodesk/")[1])]
        
        actions = f['data'][demo_id]['actions'].astype('float32')
        # frames = f['data'][demo_id]['obs']['camera_image']# [start_step:end_step].transpose(0, 3, 1, 2)

        frames_list = np.load(os.path.join(self.videos_path, sample['file'].split('robodesk/')[1][:-5], demo_id+'.npy')).transpose(0, 2, 3, 1)
        frames_list = [frame[..., ::-1] for frame in frames_list]

        return frames_list[:self.seq_length], actions[:self.seq_length-1]
            
    def __getitem__(self, index):
        imgs, actions = self.getimg(index)
        imgs = self.aug_seq(imgs, 224, 224) # seq len 35
        length = len(imgs)
        # if random.randint(0, 1):
        #     for i in range(length):
        #         imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_CLOCKWISE)
        # elif random.randint(0, 1):
        #     for i in range(length):
        #         imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_180)
        # elif random.randint(0, 1):
        #     for i in range(length):
        #         imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        # if random.uniform(0, 1) < 0.5:
        #     for i in range(length):
        #         imgs[i] = imgs[i][:, :, ::-1]
        # if random.uniform(0, 1) < 0.5:
        #     for i in range(length):
        #         imgs[i] = imgs[i][::-1]
        # if random.uniform(0, 1) < 0.5:
        #     for i in range(length):
        #         imgs[i] = imgs[i][:, ::-1]
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)
        imgs = torch.stack(imgs, 0)

        return imgs, actions#n, c, h , w
    

class RobodeskValDataset(Dataset):
    def __init__(self):
        self.path = './data/robodesk/'

        self.seq_length = 9

        self.videos_path = os.path.join(self.path, 'videos')

        self.index_path=os.path.join(self.path, 'robodesk_v2.json')

        with open(self.index_path, 'r') as f:
            self.train_data = json.load(f)

        random.shuffle(self.train_data)

        all_files = np.unique([self.train_data[i]['file'] for i in range(len(self.train_data))])


        self.all_files = {}
        for file in all_files:
            file = os.path.join(self.path, file.split("/robodesk/")[1])
            self.all_files[file] = h5py.File(file, 'r')

    def __len__(self):
        return 100# len(self.train_data)       
    
    def getimg(self, index):
        sample = self.train_data[index]
        demo_id = sample['demo']
        
        f = self.all_files[os.path.join(self.path, sample['file'].split("/robodesk/")[1])]
        
        actions = f['data'][demo_id]['actions'].astype('float32')
        # frames = f['data'][demo_id]['obs']['camera_image']# [start_step:end_step].transpose(0, 3, 1, 2)

        frames_list = np.load(os.path.join(self.videos_path, sample['file'].split('robodesk/')[1][:-5], demo_id+'.npy')).transpose(0, 2, 3, 1)
        frames_list = [frame[..., ::-1] for frame in frames_list]
        data_name = sample['file'].split('/')[-2] + '_' + sample['demo']
        return frames_list[:self.seq_length], actions[:self.seq_length-1], data_name
            
    def __getitem__(self, index):
        imgs, actions, name = self.getimg(index)
        length = len(imgs)
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)
        imgs = torch.stack(imgs, 0)
        return imgs, actions, name#n, c, h , w

def dataset2path(dataset_name):
    if dataset_name == 'robo_net':
        version = '1.0.0'
    elif dataset_name == 'language_table':
        version = '0.0.1'
    else:
        version = '0.1.0'
    return f'./data/rtx/{dataset_name}/{version}'

def episode2steps(episode):
    return episode['steps']

def step_map_fn(step):
    return {
        'observation': {
            'image': tf.image.resize(step['observation']['image'], (256, 256)),
        },
        
        'action': tf.concat([
            step['action']['world_vector'],
            step['action']['rotation_delta'],
            step['action']['gripper_closedness_action'],
        ], axis=-1)
    }

def combine_lists(*lists,):
    combined = []
    for lst in lists:
        groups = [lst[i:i + 9] for i in range(0, len(lst) - len(lst) % 9, 9)]
        groups.append(lst[-9:])
        combined.extend(groups)
    return combined

class RTXTrainDataset(Dataset):
    def __init__(self):
        DATASETS = [
            "columbia_cairlab_pusht_real"
        ]
        
        # datasets = ['columbia_cairlab_pusht_real', 'berkeley_autolab_ur5']
        datasets = ['columbia_cairlab_pusht_real']
        self.all_frames = []
        self.all_actions = []


        
        for subset in datasets:
            # if not os.path.exists('./data/rtx/%s_train.pth' % subset) and not os.path.exists('./data/rtx/%s_train_0.pth' % subset):
            if True:
                b = tfds.builder_from_directory("/data/rtx/columbia_cairlab_pusht_real/0.1.0")
                ds = b.as_dataset(split='train')
                subset_frames, subset_actions = self.subset_processing(ds, subset)
                torch.save('./data/rtx/%s_train.pth' % subset, {'all_frames': subset_frames, 'all_actions': subset_actions, 'name': subset, 'split': 'train'})
            else:
                if subset == 'columbia_cairlab_pusht_real':
                    data = torch.load('./data/rtx/%s_train.pth' % subset)
                    subset_frames = data['all_frames']
                    subset_actions = data['all_actions']
                elif subset == 'berkeley_autolab_ur5':
                    subset_frames = []
                    subset_actions = []
                    for file in os.listdir('./data/rtx'):
                        if 'berkeley_autolab_ur5_train_' in file:
                            data = torch.load('./data/rtx/%s' % file)
                            subset_frames.extend(data['all_frames'])
                            subset_actions.extend(data['all_actions'])

            self.all_frames.extend(subset_frames)
            self.all_actions.extend(subset_actions)
        

        # for subset in datasets:
        #     b = tfds.builder_from_directory(builder_dir=dataset2path(subset))
        #     ds = b.as_dataset(split='train')
        #     subset_frames, subset_actions = self.subset_processing(ds, subset)

    def resize_frames(self, frames):

        resized_frames = np.empty((frames.shape[0], frames.shape[1], 256, 256, 3), dtype=np.uint8)

        for i in range(frames.shape[0]):
            for j in range(frames.shape[1]):

                frame = frames[i, j]

                img = Image.fromarray(frame)

                img_cropped = img.crop((40, 0, 280, 240))

                img_resized = img_cropped.resize((256, 256))

                resized_frames[i, j] = np.array(img_resized)
        return resized_frames

    def subset_processing(self, ds, subset_name):
        
        if subset_name  == 'columbia_cairlab_pusht_real':
            subset_frames = []
            subset_actions = []
            for episode in tqdm(iter(ds),):
                frames = [step['observation']['image'].numpy() for step in episode['steps']]
                gripper_closedness_actions = np.array([step['action']['gripper_closedness_action'].numpy() for step in episode['steps']])
                world_vector_actions = np.array([step['action']['world_vector'].numpy() for step in episode['steps']])
                rotation_delta = np.array([step['action']['rotation_delta'].numpy() for step in episode['steps']])

                actions = np.concatenate([world_vector_actions, rotation_delta, gripper_closedness_actions[:, np.newaxis]], 1)
                import ipdb;ipdb.set_trace()
                frames = np.array(combine_lists(frames))
                frames = self.resize_frames(frames)
                actions = np.array(combine_lists(actions))

                subset_frames.extend(frames[..., ::-1])
                subset_actions.extend(actions[:,:-1])
                import ipdb;ipdb.set_trace()
        return subset_frames, subset_actions
    
    def aug_seq(self, imgs, h, w):
        ih, iw, _ = imgs[0].shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x+h, y:y+w, :]
        return imgs
    
    def __len__(self,):
        return len(self.all_frames)
    
    def __getitem__(self, idx):
        imgs, actions = self.all_frames[idx], self.all_actions[idx]

        imgs = self.aug_seq(imgs, 256, 256)

        torch_imgs = []
        length = len(imgs)

        for i in range(length):
            # imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)
            torch_imgs.append(torch.from_numpy(imgs[i].copy()).permute(2, 0, 1))
        imgs = torch.stack(torch_imgs, 0)

        return imgs, actions#n, c, h , w



    
class RTXValDataset(Dataset):
    def __init__(self):
        DATASETS = [
            "columbia_cairlab_pusht_real"
        ]
        
        self.datasets = ['columbia_cairlab_pusht_real','berkeley_autolab_ur5']
        
        
        self.all_frames = []
        self.all_actions = []

        
        for subset in self.datasets:
            if not os.path.exists('./data/rtx/%s_test.pth' % subset):
                b = tfds.builder_from_directory(builder_dir=dataset2path(subset))
                ds = b.as_dataset(split='test')
                subset_frames, subset_actions = self.subset_processing(ds, subset)
                data = {'all_frames': subset_frames, 'all_actions': subset_actions, 'name': subset, 'split': 'test'}
                torch.save(data, './data/rtx/%s_test.pth' % subset)
            else:   
                data = torch.load('./data/rtx/%s_test.pth' % subset)
                subset_frames = data['all_frames']
                subset_actions = data['all_actions']
            self.all_frames.extend(subset_frames)
            self.all_actions.extend(subset_actions)

        # ds = ds.map(episode2steps, num_parallel_calls=tf.data.AUTOTUNE).flat_map(lambda x: x)
        # ds = ds.map(step_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    def resize_frames(self, frames):

        resized_frames = np.empty((frames.shape[0], frames.shape[1], 256, 256, 3), dtype=np.uint8)

        for i in range(frames.shape[0]):
            for j in range(frames.shape[1]):

                frame = frames[i, j]

                img = Image.fromarray(frame)

                img_cropped = img.crop((40, 0, 280, 240))

                img_resized = img_cropped.resize((256, 256), Image.ANTIALIAS)

                resized_frames[i, j] = np.array(img_resized)
        return resized_frames

    def subset_processing(self, ds, subset_name):
        
        if subset_name  == 'columbia_cairlab_pusht_real':
            subset_frames = []
            subset_actions = []
            for episode in tqdm(iter(ds),):
                frames = [step['observation']['image'].numpy() for step in episode['steps']]
                gripper_closedness_actions = np.array([step['action']['gripper_closedness_action'].numpy() for step in episode['steps']])
                world_vector_actions = np.array([step['action']['world_vector'].numpy() for step in episode['steps']])
                rotation_delta = np.array([step['action']['rotation_delta'].numpy() for step in episode['steps']])

                actions = np.concatenate([world_vector_actions, rotation_delta, gripper_closedness_actions[:, np.newaxis]], 1)
                
                frames = np.array(combine_lists(frames))
                frames = self.resize_frames(frames)
                actions = np.array(combine_lists(actions))

                subset_frames.extend(frames[..., ::-1])
                subset_actions.extend(actions[:,:-1])
        return subset_frames, subset_actions
    
    def aug_seq(self, imgs, h, w):
        ih, iw, _ = imgs[0].shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x+h, y:y+w, :]
        return imgs
    
    def __len__(self,):
        return len(self.all_frames)
    
    def __getitem__(self, idx):
        imgs, actions = self.all_frames[idx], self.all_actions[idx]

        imgs = self.aug_seq(imgs, 256, 256)

        torch_imgs = []
        length = len(imgs)

        for i in range(length):
            # imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)
            torch_imgs.append(torch.from_numpy(imgs[i].copy()).permute(2, 0, 1))
        imgs = torch.stack(torch_imgs, 0)
        name = '%s_demo_%s' % (self.datasets[0], idx)
        return imgs, actions, name#n, c, h , w
    

class gif_saving(Dataset):
    def __init__(self, batch_size=16):
        builder = tfds.builder_from_directory('/data/language_table/language_table_sim/0.0.1')
        ds = builder.as_dataset(split='train[:50000]',shuffle_files=True)
        subset_frames, subset_actions = self.subset_processing(ds)
        
        # for i in range(len(subset_actions)):
        #     imageio.mimsave(f'/data/language_table/language_table_sim_gifs/video_{i}.gif', subset_frames[i], 'GIF', duration = 0.5)
        #     torch.save(subset_actions, f"/data/language_table/language_table_sim_gifs/action_{i}.pth")
            
            
    def subset_processing(self, ds):
        subset_frames = []
        subset_actions = []
        i = 0
        for episode in tqdm(iter(ds)):
            frames = [step['observation']['rgb'].numpy() for step in episode['steps']]
            actions = [step['action'].numpy() for step in episode['steps']]
            
            if len(frames) < 9:
                continue
            
            frames = np.array(combine_lists(frames))
            frames = self.resize_frames(frames)
            actions = np.array(combine_lists(actions))
            
            for j in range(frames.shape[0]):
                import ipdb;ipdb.set_trace()
                imageio.mimsave(f'/data/language_table/language_table_sim_gifs_shuffle3/video_{i+j}.gif', frames[j], 'GIF', duration = 0.5)
                torch.save(actions[j], f"/data/language_table/language_table_sim_gifs_shuffle3/action_{i+j}.pth")
                
            i = i + frames.shape[0]

            # subset_frames.extend(frames[..., ::-1])
            # subset_actions.extend(actions[:,:-1])
        return subset_frames, subset_actions
        # self.episode_ds = self.episode_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
    def resize_frames(self, frames):

        resized_frames = np.empty((frames.shape[0], frames.shape[1], 180, 320, 3), dtype=np.uint8)

        for i in range(frames.shape[0]):
            for j in range(frames.shape[1]):

                frame = frames[i, j]

                img_resized = cv2.resize(frame,(320,180))
                
                resized_frames[i, j] = np.array(img_resized)
        return resized_frames

    
    

    
class LanguageTableTrainDataset(Dataset):
    def __init__(self, batch_size=16):
        builder = tfds.builder_from_directory('./data/language_table_sim/0.0.1/')
        self.episode_ds = builder.as_dataset(split='train')
        self.episode_ds = self.episode_ds.shuffle(buffer_size=8)
        # self.episode_ds = self.episode_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def __len__(self,):
        return len(self.episode_ds)
    
    def __getitem__(self, idx):
        episode = next(iter(self.episode_ds.skip(idx).take(1)))

        frames = []
        actions = []
        
        count = 0
        for step in episode['steps'].as_numpy_iterator():
            
            if len(episode['steps']) >= 9:
                frames.append(torch.from_numpy(cv2.resize(step['observation']['rgb'], (320, 180), interpolation=cv2.INTER_AREA)))
                actions.append(torch.from_numpy(step['action']))
                count += 1
                
                if count == 9:
                    break
            else:
                if count == 0:
                    for _ in range(9 - len(episode['steps'])):
                        frames.append(torch.from_numpy(cv2.resize(step['observation']['rgb'], (320, 180), interpolation=cv2.INTER_AREA)))
                        actions.append(torch.from_numpy(step['action']))
                frames.append(torch.from_numpy(cv2.resize(step['observation']['rgb'], (320, 180), interpolation=cv2.INTER_AREA)))
                actions.append(torch.from_numpy(step['action']))
                count += 1

        imgs = torch.stack(frames, 0).permute(0, 3, 1, 2)

        actions = torch.stack(actions, 0)

        return imgs, actions
    


class LanguageTableValDataset(Dataset):
    def __init__(self, batch_size=16):
        builder = tfds.builder_from_directory('./data/language_table_sim/0.0.1/')
        self.episode_ds = builder.as_dataset(split='train')
        self.episode_ds = self.episode_ds.shuffle(buffer_size=8)
        # self.episode_ds = self.episode_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def __len__(self,):
        return 2000# len(self.episode_ds)
    
    def __getitem__(self, idx):
        episode = next(iter(self.episode_ds.skip(idx).take(1)))

        frames = []
        actions = []
        
        count = 0
        for step in episode['steps'].as_numpy_iterator():
            
            if len(episode['steps']) >= 9:
                frames.append(torch.from_numpy(cv2.resize(step['observation']['rgb'], (320, 180), interpolation=cv2.INTER_AREA)))
                actions.append(torch.from_numpy(step['action']))
                count += 1
                
                if count == 9:
                    break
            else:
                if count == 0:
                    for _ in range(9 - len(episode['steps'])):
                        frames.append(torch.from_numpy(cv2.resize(step['observation']['rgb'], (320, 180), interpolation=cv2.INTER_AREA)))
                        actions.append(torch.from_numpy(step['action']))
                frames.append(torch.from_numpy(cv2.resize(step['observation']['rgb'], (320, 180), interpolation=cv2.INTER_AREA)))
                actions.append(torch.from_numpy(step['action']))
                count += 1

        imgs = torch.stack(frames, 0)
        actions = torch.stack(actions, 0)

        return imgs, actions

        

    

if __name__ == '__main__':
    
    dataset = gif_saving()
    # dataset = RTXTrainDataset()
    
    # language_table_train_dataset = LanguageTableTrainDataset()
    
    # train_data = DataLoader(language_table_train_dataset, batch_size=16, num_workers=0, pin_memory=True)
    # for data in tqdm(train_data):
    #     pass
    # import ipdb; ipdb.set_trace()

    
    # def tf_collate_fn(batch):
    #     x, y = zip(*batch)
    #     x = torch.stack(x).permute(0, 3, 1, 2).type(torch.FloatTensor)
    #     y = torch.stack(y)
    #     return x, y
    
    # def iter_tf_data(train_ds):
    #     x_list = []
    #     y_list = []
    #     for data in train_ds.as_numpy_iterator():
    #         x, y = data
    #         x_list += [torch.from_numpy(x)] 
    #         y_list += [torch.from_numpy(y)]
    #     x_list_cat = torch.cat(x_list, axis=0)
    #     y_list_cat = torch.cat(y_list, axis=0)
    #     return [x_list_cat, y_list_cat]

    # for i in range(len(language_table_train_dataset)):
    #     import ipdb; ipdb.set_trace()
    #     a = language_table_train_dataset[i]
    # imgs, actions = rtx_train_dataset[-1][0], rtx_train_dataset[-1][1]
    # import ipdb;ipdb.set_trace()
    
    
    # from PIL import Image, ImageDraw, ImageFont
    # def add_text_to_image(image, text, position=(0, 0), font_size=10):
    #     """
    #     在PIL图像上添加文字
    #     """
    #     draw = ImageDraw.Draw(image)
    #     font = ImageFont.load_default()
    #     draw.text(position, text, font=font, fill=(255, 255, 255),font_size = font_size)
    #     return image

    # for j in range(10):
    #     imgs, actions = rtx_train_dataset[-j-50][0], rtx_train_dataset[-j-50][1]
    #     print(actions)
        
    #     frames = []
    #     for i in range(imgs.shape[0]):
    #         # 转换为 PIL 图像
    #         pil_img = Image.fromarray(imgs[i].numpy().astype(np.uint8).transpose(1, 2, 0))
    #         # 在图像上添加文字
    #         if i != 8:
    #             # import ipdb;ipdb.set_trace()
    #             pil_img = add_text_to_image(pil_img, str(actions[i]), position=(10, 10))
    #         # 添加到帧列表中
    #         frames.append(pil_img)
            
    #     frames[0].save(f'output_ber{j}.gif', format='GIF', append_images=frames[1:], save_all=True, duration=1000, loop=0)