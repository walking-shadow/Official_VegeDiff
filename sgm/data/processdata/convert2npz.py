import os
import datetime
import torchvision
import pytorch_lightning as pl
import numpy as np
import kornia
import kornia.augmentation as K
import json
import random
import torch
import argparse

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sgm.utils.videoio_util import *
from sgm.utils.ceph_util import PetrelBackend


from sgm.utils.videoio_util import PetrelVideo, numpy_array_to_video, PetrelDecordReader
from omegaconf import OmegaConf
from sgm.data.webvid10m import Webvid10MLoader
from tqdm import tqdm

class WebvidDataDictWrapper(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, i):
        results = self.dataset[i]
        return {"txt": results['txt'],
                "imgs": results['imgs'],
                "original_size_as_tuple": results['original_size_as_tuple'],
                "crop_coords_top_left": results['crop_coords_top_left'],
                "target_size_as_tuple": results['target_size_as_tuple']}

    def __len__(self):
        return self.dataset.__len__()

class Webvid10MDataset(Dataset):
    def __init__(self, 
                 data_path, 
                 meta_path, 
                 clip_length, 
                 clip_FPS_reate, 
                 npzsuffix, 
                 num_threads=0, 
                 resize_resolution=(256, 256), 
                 crop_resolution=(256, 256), 
                 horizontal_flip=0.5, 
                 video_reader_choice='original',
                 ):
        # path
        self.data_path = data_path
        self.meta_path = meta_path
        
        self.npzsuffix = npzsuffix
        
        # augmentation arguments
        self.resize_resolution = resize_resolution
        self.crop_resolution = crop_resolution
        self.horizontal_flip = horizontal_flip
        
        # sampler arguments
        self.clip_length = clip_length
        self.clip_FPS_reate = clip_FPS_reate
        
        # configure petrel-oss and videoreader
        self.petrel_backend=PetrelBackend()
        self.video_reader_choice = video_reader_choice
        if self.video_reader_choice == 'original':
            self.video_reader=PetrelVideoReader(file_client=self.petrel_backend)
        elif self.video_reader_choice == 'petrel_original':
            self.video_reader=PetrelVideo(file_client=self.petrel_backend)
        elif self.video_reader_choice == 'petrel_decord':
            self.video_reader=PetrelDecordReader(file_client=self.petrel_backend)
        # read path list
        with open(self.meta_path, "r") as file:
            self.path_list = file.readlines()
        self.path_list = [line.strip() for line in self.path_list]
        
    def video_aug(self, results, resize_resolution, crop_resolution, horizontal_flip):
        frames_tensor = np.asarray_chkfinite(results['imgs'], dtype=np.uint8).transpose(0,3,1,2) #TCHW
        # if self.video_reader_choice == 'original' or self.video_reader_choice == 'petrel_decord':
        #     frames_tensor = (kornia.image_to_tensor(frames_tensor, keepdim=False)*2.0-255.0).div(255.0)
        # elif self.video_reader_choice == 'petrel_original':
        #     frames_tensor = (torch.from_numpy(frames_tensor)*2.0-255.0).div(255.0)
        # T, C, H, W = frames_tensor.shape# (x*2-255)/255
        # frames_tensor_ = frames_tensor.view(1, T, C, H, W)
        # # resize, centercrop, horizontalflip
        # if len(resize_resolution)==1:
        #     resize_resolution = resize_resolution[0]
        # aug_list = K.VideoSequential(
        #     kornia.augmentation.Resize(resize_resolution),
        #     kornia.augmentation.CenterCrop(crop_resolution),
        #     kornia.augmentation.RandomHorizontalFlip(p=horizontal_flip),
        #     data_format="BTCHW",
        #     same_on_frame=True)
        # frames_tensor = aug_list(frames_tensor_)
        results['imgs'] = frames_tensor
        results['img_shape'] = results['imgs'][0].shape
        return results
    
    def get_results(self, idx):
        
        read_path = self.path_list[idx]
        video_path = os.path.join(self.data_path, read_path+'.mp4')
        json_path = os.path.join(self.data_path, read_path+'.json')
        json_data = self.petrel_backend.get_json(filepath=json_path)
        save_dir = os.path.join(self.npzsuffix, 
                                read_path.split("/")[-3], 
                                read_path.split("/")[-2], 
                                read_path.split("/")[-1])
        if self.petrel_backend.exists(filepath=os.path.join(save_dir,"all_frames.npz")) and self.petrel_backend.exists(filepath=os.path.join(save_dir, "meta.json")):
            try:
                json_break_data = self.petrel_backend.get_json(filepath=os.path.join(save_dir, "meta.json"))
                if "frameinfo" in json_break_data.keys():
                    return None
                else:
                    pass
            except (RuntimeError, Exception, BaseException, TypeError) as e:
                self.petrel_backend.remove(filepath=os.path.join(save_dir, "meta.json"))

            # json_break_data = self.petrel_backend.get_json(filepath=os.path.join(save_dir, "meta.json"))
            # if "frameinfo" in json_break_data.keys():
            #     return None
        results = self.video_reader.sample_whole_video(filename=video_path)
        results = self.video_aug(results, self.resize_resolution, self.crop_resolution, self.horizontal_flip)
        '''
        s3://infdata/video/webvid10m0_npz/00000/00000001
            meta.json
            all_frames.npz
        '''
        if self.petrel_backend.exists(filepath=os.path.join(save_dir,"all_frames.npz")):
            self.petrel_backend.remove(filepath=os.path.join(save_dir,"all_frames.npz"))
        self.petrel_backend.put_npz(filepath=os.path.join(save_dir,"all_frames.npz"), value=results['imgs'])
        
        if self.petrel_backend.exists(filepath=os.path.join(save_dir, "meta.json")):
            self.petrel_backend.remove(filepath=os.path.join(save_dir, "meta.json"))
        json_data["frameinfo"] = {}
        json_data["frameinfo"]["total_frames"] = results["total_frames"]
        json_data["frameinfo"]["avg_fps"] = results['avg_fps']
        json_data["frameinfo"]["img_shape"] = results['img_shape']
        self.petrel_backend.put_json(filepath=os.path.join(save_dir, "meta.json"), value=json_data)
        
        results['img_shape'] = results['imgs'].shape
        results['txt'] = json_data["caption"]
        results['original_size_as_tuple'] = torch.tensor([0, 0])
        results['crop_coords_top_left'] = torch.tensor([results['img_shape'][-2], results['img_shape'][-1]])
        results['target_size_as_tuple'] = torch.tensor([results['img_shape'][-2], results['img_shape'][-1]])
        return results
    
    def __len__(self,):
        return len(self.path_list)
    
    def __getitem__(self, idx):
        results = {}
        if self.video_reader_choice == 'original':
            while(True):
                try:
                    results = self.get_results(idx)
                    break
                except (RuntimeError, Exception, BaseException) as e:
                    print(e, " : ", self.path_list[idx])
                    idx = random.randint(0,self.__len__()-1)
        elif self.video_reader_choice == 'petrel_original':
            print(self.path_list[idx])
            try:
                results = self.get_results(idx)
            except (RuntimeError, Exception, BaseException, TypeError) as e:
                print(e, " : ", self.path_list[idx])
                with open("webvid10m_error_meta.txt", "a+") as file:
                    file.write(str(self.path_list[idx])+"\n")
                results = {"txt": "None",
                           "imgs": torch.zeros([self.clip_length, 3, self.crop_resolution[0], self.crop_resolution[1]]),
                           "original_size_as_tuple": torch.tensor([0, 0]),
                           "crop_coords_top_left": torch.tensor([0, 0]),
                           "target_size_as_tuple": torch.tensor([0, 0])}
        elif self.video_reader_choice == 'petrel_decord':
            print(self.path_list[idx])
            try:
                _ = self.get_results(idx)
                _ = None
                results = {"txt": "None",
                           "imgs": torch.zeros([self.clip_length, 3, self.crop_resolution[0], self.crop_resolution[1]]),
                           "original_size_as_tuple": torch.tensor([0, 0]),
                           "crop_coords_top_left": torch.tensor([0, 0]),
                           "target_size_as_tuple": torch.tensor([0, 0])}
            except (RuntimeError, Exception, BaseException, TypeError) as e:
                print(e, " : ", self.path_list[idx])
                results = {"txt": "None",
                           "imgs": torch.zeros([self.clip_length, 3, self.crop_resolution[0], self.crop_resolution[1]]),
                           "original_size_as_tuple": torch.tensor([0, 0]),
                           "crop_coords_top_left": torch.tensor([0, 0]),
                           "target_size_as_tuple": torch.tensor([0, 0])}
        return results

# Full Clip:
def init_dataloader(video_reader_choice, metadir, npzsuffix):
    train_dataset = WebvidDataDictWrapper(
        Webvid10MDataset(
            data_path="s3://infdata/video/", 
            meta_path=metadir, 
            clip_length=16, 
            clip_FPS_reate=4,
            resize_resolution=tuple([256]), 
            crop_resolution=tuple([256, 256]), 
            horizontal_flip=0,
            video_reader_choice=video_reader_choice,
            npzsuffix=npzsuffix
        )
    )
    return DataLoader(
                train_dataset,
                batch_size=16,
                num_workers=16,
                shuffle=False,
                pin_memory=False,
                drop_last=False
            )


# split 64
def parse_args():
    parser = argparse.ArgumentParser(description="Argument.")
    parser.add_argument(
        "--project_name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--npzsuffix",
        type=str,
        default="s3://infdata/video/webvid10m_npz",
    )
    parser.add_argument(
        "--metadir",
        type=str,
        default="/mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_meta.txt",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    webvid_dataloader=init_dataloader(video_reader_choice="petrel_decord",
                                      metadir=args.metadir,
                                      npzsuffix=args.npzsuffix)
    idx=0
    print("metadir: ", args.metadir)
    print("npzsuffix: ", args.npzsuffix)
    progress_bar = tqdm(len(webvid_dataloader))
    progress_bar.set_description(args.metadir + "-progress")
    for i in tqdm(webvid_dataloader):
        progress_bar.update(1)
        idx+=1


if __name__ == "__main__":
    main()