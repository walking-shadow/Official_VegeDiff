import os
import datetime
import torchvision
import pytorch_lightning as pl
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class ImagenetDataDictWrapper(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, i):
        x, y = self.dataset[i]
        return {"jpg": x, "cls": y}

    def __len__(self):
        return len(self.dataset)


# class CustomDataset(StreamingVisionDataset):
#     def __init__(self, local, remote, **kwargs):
#         super().__init__(local=local, remote=remote, **kwargs)   

#     def get_item(self, idx):
#         obj = super().__getitem__(idx)
#         return obj[0], obj[1]
class CustomDataset(Dataset):
    def __init__(self, local, remote, transform, **kwargs):
        super().__init__(local=local, remote=remote, **kwargs)
        self.transform = transform

    def __getitem__(self, idx):
        obj = super().__getitem__(idx)
        # assert obj['x'].shape[0] == 3
        # if obj['x'].mode=='L':
        #     print(obj['x'].mode, idx)
        #     return obj['x'], obj['y']
        # return self.transform(obj['x']), obj['y']
        return obj[0], obj[1]


class ImagenetLoader(pl.LightningDataModule):
    def __init__(self,
                 train):
        super().__init__()

        self.train_config = train

        self.batch_size = self.train_config.loader.batch_size
        self.num_workers = self.train_config.loader.num_workers
        self.shuffle = self.train_config.loader.shuffle

        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, self.train_config.resize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.train_dataset = ImagenetDataDictWrapper(ImageFolder(self.train_config.data_path, transform=transform))
        
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

if __name__ == "__main__":
    from omegaconf import OmegaConf
    conf = OmegaConf.load('/home/luzeyu/projects/workspace/generative-models/configs/example_training/dataset/imagenet-256-streaming.yaml')
    indataloader=ImagenetLoader(train=conf.data.params.train).train_dataloader()
    from tqdm import tqdm
    for i in tqdm(indataloader):
        # print(i)
        # print("*"*20)
        pass