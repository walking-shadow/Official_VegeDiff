import os
import datetime
import torchvision
import numpy as np
import kornia
import kornia.augmentation as K
import json
import random
import torch

from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sgm.utils.videoio_util import *
from sgm.utils.ceph_util import PetrelBackend
from pathlib import Path
from petrel_client.client import Client
import xarray as xr


'''
Webvid dataset:

mp4->BTCHW
FPS: frames per second

arguments: 
1. data_root
2. annotation_path
3. resize_resolution
4. horizontal_flip
5. clip_length
6. clip_FPS_reate

read(clip_length, clip_FPS_reate): clip_length, clip_FPS_reate -> frame_id_list
transform
read_others

return:
dict([
    'filename', 
    'total_frames', 
    'video_reader', 
    'avg_fps', 
    'frame_inds', 
    'imgs', 
    'original_shape', 
    'img_shape',
    'txt',
    'original_size_as_tuple',
    'crop_coords_top_left',
    'target_size_as_tuple'
])
'''

class WebvidDataDictWrapper(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, i):
        results = self.dataset[i]
        return {
                "imgs": results['imgs'],
                "original_size_as_tuple": results['original_size_as_tuple'],
                "crop_coords_top_left": results['crop_coords_top_left'],
                "target_size_as_tuple": results['target_size_as_tuple']}

    def __len__(self):
        return self.dataset.__len__()


class Webvid2MDataset(Dataset):
    def __init__(self, data_path, meta_path, clip_length, clip_FPS_reate, num_threads=0, resize_resolution=(256, 256), crop_resolution=(256, 256), horizontal_flip=0.5):
        # path
        self.data_path = data_path
        self.meta_path = meta_path
        
        # augmentation arguments
        self.resize_resolution = resize_resolution
        self.crop_resolution = crop_resolution
        self.horizontal_flip = horizontal_flip
        
        # sampler arguments
        self.clip_length = clip_length
        self.clip_FPS_reate = clip_FPS_reate
        
        # configure petrel-oss and videoreader
        self.petrel_backend=PetrelBackend()
        self.video_reader=PetrelVideoReader(file_client=self.petrel_backend)

        # read path list
        with open(self.meta_path, "r") as file:
            self.path_list = file.readlines()
        self.path_list = [line.strip() for line in self.path_list]
        
    def video_aug(self, results, resize_resolution, crop_resolution, horizontal_flip):
        frames_tensor = np.asarray_chkfinite(results['imgs'], dtype=np.uint8)
        frames_tensor = kornia.image_to_tensor(frames_tensor, keepdim=False).div(255.0)
        T, C, H, W = frames_tensor.shape
        frames_tensor_ = frames_tensor.view(1, T, C, H, W)
        # resize, centercrop, horizontalflip
        aug_list = K.VideoSequential(
            kornia.augmentation.Resize(resize_resolution),
            kornia.augmentation.CenterCrop(crop_resolution),
            kornia.augmentation.RandomHorizontalFlip(p=horizontal_flip),
            data_format="BTCHW",
            same_on_frame=True)
        frames_tensor = aug_list(frames_tensor_)
        _, T, C, H, W = frames_tensor.shape
        results['imgs'] = frames_tensor.view(T, C, H, W)
        results['img_shape'] = results['imgs'][0].shape
        return results # imgs:BTCHW
    
    def get_results(self, idx):
        read_path = self.path_list[idx]
        video_path = os.path.join(self.data_path, read_path+'.mp4')
        json_path = os.path.join(self.data_path, read_path+'.json')
        json_data = self.petrel_backend.get_json(filepath=json_path)
        results = self.video_reader.sample_clip(clip_length=self.clip_length, clip_FPS_reate=self.clip_FPS_reate, filename=video_path)
        results = self.video_aug(results, self.resize_resolution, self.crop_resolution, self.horizontal_flip)
        results['img_shape'] = results['imgs'].shape
        results['txt'] = json_data["caption"]
        results['original_size_as_tuple'] = torch.tensor([0, 0])
        results['crop_coords_top_left'] = torch.tensor([results['img_shape'][-2], results['img_shape'][-1]])
        results['target_size_as_tuple'] = torch.tensor([results['img_shape'][-2], results['img_shape'][-1]])
        return results
    
    def __len__(self,):
        return len(self.path_list)
    
    def __getitem__(self, idx):
        while(True):
            try:
                results = self.get_results(idx)
                break
            except Exception as e:
                print(Exception)
                print("error:", self.path_list[idx])
                idx = random.randint(0,self.__len__()-1)
        return results


class Webvid2MLoader():
    def __init__(self, train_config):
        super().__init__()
        self.train_config = train_config
        self.batch_size = self.train_config.loader.batch_size
        self.num_workers = self.train_config.loader.num_workers
        self.shuffle = self.train_config.loader.shuffle
        self.train_dataset = WebvidDataDictWrapper(Webvid2MDataset(
            data_path=self.train_config.data_path, 
            meta_path=self.train_config.meta_path, 
            clip_length=self.train_config.clip_length, 
            clip_FPS_reate=self.train_config.clip_FPS_reate,
            resize_resolution=tuple(self.train_config.resize_resolution), 
            crop_resolution=tuple(self.train_config.crop_resolution), 
            horizontal_flip=self.train_config.horizontal_flip
            ))
        self.test_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def test_dataloader(self):
        return None

    def val_dataloader(self):
        return None


class Webvid10MDataset(Dataset):
    def __init__(self, 
                folder: str, 
                data_path_file,
                fp16 = False,
                s2_bands = ["B02", "B03", "B04", "B8A"], 
                eobs_vars = ['fg', 'hu', 'pp', 'qq', 'rr', 'tg', 'tn', 'tx'], 
                eobs_agg = ['mean', 'min', 'max'], 
                static_vars = ['nasa_dem', 'alos_dem', 'cop_dem', 'esawc_lc', 'geom_cls'], 
                start_month_extreme = None, 
                dl_cloudmask = True, 
                min_lc = 10,
                max_lc = 40,
                noise_image_num=20,
                # high_resolution = [128, 128],
                # meso_resolution = [80, 80]
                #
                ):
        
        # zsj: earthnet2021参数
        # if not isinstance(folder, Path):
        #     folder = Path(folder)
        # assert (not {"target","context"}.issubset(set([d.name for d in folder.glob("*") if d.is_dir()])))
        # self.filepaths = sorted(list(folder.glob("**/*.npz"))) 
        conf_path = '~/petreloss.conf'
        self.client = Client(conf_path) # 若不指定 conf_path ，则从 '~/petreloss.conf' 读取配置文件
        with open(data_path_file, 'r') as file:
            self.filepaths = [folder+'/'+line.strip() for line in file]

        self.type = np.float16 if fp16 else np.float32
        self.s2_bands = s2_bands
        self.eobs_vars = eobs_vars
        self.eobs_agg = eobs_agg
        self.static_vars = static_vars
        self.start_month_extreme = start_month_extreme
        self.dl_cloudmask = dl_cloudmask
        self.min_lc = min_lc
        self.max_lc = max_lc
        self.noise_image_num = noise_image_num
        # self.high_resolution = high_resolution
        # self.meso_resolution = meso_resolution


        self.eobs_mean = xr.DataArray(data = [8.90661030749754, 2.732927619847993, 77.54440854529798, 1014.330962704611, 126.47924227500346, 1.7713217310829938, 4.770701430461286, 13.567999825718509], coords = {'variable': ['eobs_tg', 'eobs_fg', 'eobs_hu', 'eobs_pp', 'eobs_qq', 'eobs_rr', 'eobs_tn', 'eobs_tx']}) 
        self.eobs_std = xr.DataArray(data = [9.75620252236597, 1.4870108944469236, 13.511387994026359, 10.262645403460999, 97.05522895011327, 4.147967261223076, 9.044987677752898, 11.08198777356161], coords = {'variable': ['eobs_tg', 'eobs_fg', 'eobs_hu', 'eobs_pp', 'eobs_qq', 'eobs_rr', 'eobs_tn', 'eobs_tx']}) 

        # 这里的esawc_lc和geom_cls的方差有更改，方差设置成了这些类别里面的最大值，从而把数据规整到0-1之间
        self.static_mean = xr.DataArray(data = [0.0, 0.0, 0.0, 0.0, 0.0], coords = {'variable': ['nasa_dem', 'alos_dem', 'cop_dem', 'esawc_lc', 'geom_cls']})
        self.static_std = xr.DataArray(data = [500.0, 500.0, 500.0, 100.0, 10.0], coords = {'variable': ['nasa_dem', 'alos_dem', 'cop_dem', 'esawc_lc', 'geom_cls']})

    def earthnet_video_aug(self, results):
        import numpy as np
        import random

        def random_augmentation(array, augmentation_flag):
            """
            对数组进行随机的数据增强：水平翻转、垂直翻转或旋转90度。
            """
            if augmentation_flag[0]:
                # 水平翻转
                array = [np.flip(arr, axis=-1) for arr in array]
                    
            if augmentation_flag[1]:
                # 垂直翻转
                array = [np.flip(arr, axis=-2) for arr in array]
            if augmentation_flag[2]:
                # 旋转90度
                array = [np.rot90(arr, k=1, axes=(-2, -1)) for arr in array]
            return array
        
        def to_tensor(array):
            # 转为tensor，并且把0-1的值域变成-1到1
            t,c,h,w = array.shape
            array = np.transpose(array.reshape(t*c,h,w),(1,2,0))
            transform_image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*c*t, std=[0.5]*c*t, inplace=True)
            ])
            array = transform_image(array).view(t,c,h,w)
            return array
        
        augmentation_flag = [random.choice([0, 1]) for _ in range(3)]
        aug_array = [results['imgs'], results['mask'], results['highres_condition_image']]
        aug_array = random_augmentation(aug_array, augmentation_flag)
        results['imgs'], results['mask'], results['highres_condition_image'] = aug_array

        # 遥感图像放到-1到1的范围内
        results['imgs'] = to_tensor(results['imgs'].copy())
        # 高分辨率和低分辨率条件图像已经归一化过了，因此直接转成tensor即可
        results['highres_condition_image'] = torch.from_numpy(results['highres_condition_image'].copy())
        results['meso_condition_image'] = torch.from_numpy(results['meso_condition_image'].copy())
        # mask不进行normalize操作
        results['mask'] = torch.from_numpy(results['mask'].copy())

        return results

    
    # TODO zsj 这部分代码用于读取数据，并且会读取视频，我的视频是图片序列，这里需要修改，也需要把引导变量序列纳入进来
    def get_results(self, idx):
        # zsj: earth2021数据加载
        # TODO ZSJ 把气象数据的5天时间数据在通道上拼接起来，和遥感图像保持一致的时间序列长度
        filepath = self.filepaths[idx]
        # 改成使用client读取.npz的方式
        # npz = np.load(filepath)
        data_file = self.client.get(filepath)
        # 使用 BytesIO 来模拟一个文件
        data_file = io.BytesIO(data_file)
        # npz = np.load(data_file, allow_pickle=True)
        minicube = xr.open_dataset(data_file)


        if self.start_month_extreme:
            start_idx = {"march": 10, "april": 15, "may": 20, "june": 25, "july": 30}[self.start_month_extreme]
            minicube = minicube.isel(time = slice(5*start_idx,5*(start_idx+30)))

        # 处理RGBN遥感图像，图像大于1的部分基本都是云，因此直接规整到0-1的范围
        sen2arr = minicube[[f"s2_{b}" for b in self.s2_bands]].to_array("band").isel(time = slice(4,None,5)).transpose("time", "band", "lat", "lon").values
        sen2arr[np.isnan(sen2arr)] = 0.0 # Fill NaNs!!
        sen2arr[sen2arr<0] = 0
        sen2arr[sen2arr>1] = 1  # T,4,H,W

        # 云掩膜
        if self.dl_cloudmask:
            sen2mask = minicube.s2_dlmask.where(minicube.s2_dlmask > 0, 4*(~minicube.s2_SCL.isin([1,2,4,5,6,7]))).isel(time = slice(4,None,5)).transpose("time", "lat", "lon").values[:, None, ...]
            sen2mask[np.isnan(sen2mask)] = 4.
        else:
            sen2mask = minicube[["s2_mask"]].to_array("band").isel(time = slice(4,None,5)).transpose("time", "band", "lat", "lon").values
            sen2mask[np.isnan(sen2mask)] = 4.
        sen2mask = (sen2mask < 1.0).astype(bool) # 0值表示这个区域没有被云等因素影响，得到的结果中1值表示该区域保留  # T,1,H,W

        # 动态气象数据
        eobs = ((minicube[[f'eobs_{v}' for v in self.eobs_vars]].to_array("variable") - self.eobs_mean)/self.eobs_std).transpose("time", "variable")
        eobsarr = []
        if "mean" in self.eobs_agg:
            eobsarr.append(eobs.coarsen(time = 5, coord_func = "max").mean())
        if "min" in self.eobs_agg:
            eobsarr.append(eobs.coarsen(time = 5, coord_func = "max").min())
        if "max" in self.eobs_agg:
            eobsarr.append(eobs.coarsen(time = 5, coord_func = "max").max())
        # if "std" in self.eobs_agg:
        #     eobsarr.append(eobs.coarsen(time = 5, coord_func = "max").std())
        
        eobsarr = np.concatenate(eobsarr, axis = 1)

        eobsarr[np.isnan(eobsarr)] = 0.  # MAYBE BAD IDEA......  T,32
        # eobsarr = np.transpose(eobsarr, (1,0))  # 32，T

        # 静态环境数据
        staticarr = ((minicube[self.static_vars].to_array("variable") - self.static_mean)/self.static_std).transpose("variable", "lat", "lon").values
        staticarr[np.isnan(staticarr)] = 0.  # MAYBE BAD IDEA......  5,H,W

        # 土地分类数据
        lc = minicube[['esawc_lc']].to_array("variable").transpose("variable", "lat", "lon").values # c h w
        lc[np.isnan(lc)] = 0
        lc = ((lc >= self.min_lc).astype(bool) & (lc <= self.max_lc).astype(bool))   # 1,H,W
        lc = lc[np.newaxis, ...].repeat(sen2mask.shape[0], 0)  # T,1,H,W

        # 云掩膜与植被掩膜的汇总掩膜
        mask = (sen2mask & lc).astype(bool)  # T,1,H,W

        # 用过去图像的平均值来代替所有有云的区域
        sen2arr_past = sen2arr[:-self.noise_image_num, ...] # t1,4,h,w
        sen2mask_past = sen2mask[:-self.noise_image_num, ...].repeat(sen2arr_past.shape[1], 1)  # t1,4,h,w
        sen2arr_past = sen2arr_past * sen2mask_past
        sen2arr_past_mean = np.sum(sen2arr_past, axis=0)/(np.sum(sen2mask_past, axis=0)+1e-8) # 4,h,w
        sen2arr_past_mean = sen2arr_past_mean[np.newaxis, ...].repeat(sen2mask.shape[0], 0)  # t,c,h,w
        sen2arr = sen2arr*sen2mask + sen2arr_past_mean*(1-sen2mask)  # t,c,h,w        

        # 用所有图像中的非植被像素都替换成所有时间的非植被像素的平均值
        sen2arr_non_vege_mean = np.mean(sen2arr, axis=0, keepdims=True).repeat(sen2mask.shape[0], 0)  # t,c,h,w
        lc_mask = lc.repeat(sen2mask.shape[1], 1)  # t,c,h,w
        sen2arr = sen2arr * lc_mask + sen2arr_non_vege_mean * (1-lc_mask)

        results = {
            "imgs": sen2arr,  # T, 4, H, W
            "mask": mask,  # T, 1, H, W
            "meso_condition_image": eobsarr,  # T, 24
            "highres_condition_image": staticarr,  # 5, H, W
        }
        results = self.earthnet_video_aug(results)

        # 把图像中有云的部分通过插值使用合理的趋势值代替
        def mean_filter_with_mask(image_series, mask):
            # 输入的形状均为t,c,h,w
            # print(f'image_series:{image_series.shape},mask:{mask.shape}')

            # # 执行k次操作
            # for _ in range(k):

            image_series_pad = torch.cat([image_series[0:1,...], image_series, image_series[-1:,...]], dim=0)

            # 均值处理：对扩展后的时间序列使用3x3的均值算子
            image_series_mean = (image_series_pad[:-2,...] + image_series_pad[2:,...]) / 2

            # 替换掩膜为0的位置的值
            image_series[mask==0] = image_series_mean[mask==0]

            return image_series
        results['imgs'] = mean_filter_with_mask(results['imgs'], results['mask'].repeat(1,4,1,1))


        # results["filepath"] = str(filepath)
        # results['img_shape'] = results['imgs'].shape
        # results['original_size_as_tuple'] = torch.tensor([0, 0])
        # results['crop_coords_top_left'] = torch.tensor([results['img_shape'][-2], results['img_shape'][-1]])
        # results['target_size_as_tuple'] = torch.tensor([results['img_shape'][-2], results['img_shape'][-1]])

        return results
    
    def __len__(self,):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        results = {}
                
        while(True):
            try:
                results = self.get_results(idx)
                break
            except (RuntimeError, Exception, BaseException) as e:
                print(e, " : ", self.filepaths[idx])
                idx = random.randint(0,self.__len__()-1)
        return results
    

class Webvid10MLoader():
    # TODO zsj 这个类是用来加载数据的，可以看看怎么改
    def __init__(
        self, 
        mode,
        # high_resolution,
        # meso_resolution,
        data_root_dir,
        train_data_path_file,
        # train_data_mean,
        # train_data_var,
        fp16,
        min_lc,
        max_lc,
        noise_image_num,
        train_batch_size,
        test_batch_size,
        num_workers,
        test_track,
        ): # petrel_oiginal
        super().__init__()
        # self.config = train
        assert mode in ['train', 'test'], "mode should be 'train' or 'test'"
        self.mode = mode
        self.data_path = data_root_dir
        self.train_data_path_file = train_data_path_file
        self.fp16 = fp16
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.test_track = test_track
        self.min_lc = min_lc
        self.max_lc = max_lc
        # self.train_data_mean = train_data_mean
        # self.train_data_var = train_data_var

    # if mode == 'train':
        # 不保留验证集，全部数据都用来训练
        self.earthnet_train = Webvid10MDataset(folder=data_root_dir+'/'+'train', data_path_file = train_data_path_file,
                                            #    train_data_mean = train_data_mean, train_data_var = train_data_var,
                                            fp16=fp16, min_lc=min_lc, max_lc=max_lc, noise_image_num=noise_image_num)

        # val_size = int(val_pct * len(earthnet_corpus))
        # train_size = len(earthnet_corpus) - val_size

        # self.earthnet_train, self.earthnet_val = random_split(earthnet_corpus, [train_size, val_size], 
        #                                                       generator=torch.Generator().manual_seed(int(val_split_seed)))

        test_path_file = f'/mnt/petrelfs/zhaosijie/video_stable_diffusion/stable_diffusion_video/data_path_file/{test_track}_path_file.txt'
        self.earthnet_test = Webvid10MDataset(folder=data_root_dir+'/'+test_track, data_path_file=test_path_file,
                                                  fp16=fp16, min_lc=min_lc, max_lc=max_lc, noise_image_num=noise_image_num)



    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.earthnet_train, 
            batch_size=self.train_batch_size, 
            num_workers = self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            )
    
    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return DataLoader(
            self.earthnet_test, 
            batch_size=self.test_batch_size, 
            num_workers = self.num_workers, 
            pin_memory=True
            )


    def train_len(self):
        return len(self.earthnet_train)
    
    def test_len(self):
        return len(self.earthnet_test)




# elif key == "original_size_as_tuple":
#     batch["original_size_as_tuple"] = (
#         torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
#         .to(device)
#         .repeat(*N, 1)
#     )
# elif key == "crop_coords_top_left":
#     batch["crop_coords_top_left"] = (
#         torch.tensor(
#             [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
#         )
#         .to(device)
#         .repeat(*N, 1)
#     )
# elif key == "target_size_as_tuple":
#     batch["target_size_as_tuple"] = (
#         torch.tensor([value_dict["target_height"], value_dict["target_width"]])
#         .to(device)
#         .repeat(*N, 1)
#     )


if __name__ == "__main__":
    from omegaconf import OmegaConf
    conf = OmegaConf.load('configs/dataset/webvid10m_256.yaml')
    webvid_dataloader=Webvid2MLoader(train_config=conf.data.params.train).train_dataloader()
    # from tqdm import tqdm
    # for i in tqdm(indataloader):
    #     # print(i)
    #     # print("*"*20)
    #     pass