import os, glob
import random
import torch
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage

class LoadDataset(data.Dataset):
    def __init__(self, root, mode):
        self.imgs = sorted(glob.glob(os.path.join(root, 'blur') + "/*.jpg"))

        self.size = len(self.imgs_blur)

        tfs = []
        tfs.append(ToTensor())
        tfs.append(Normalize(mean=[0.5,], std=[0.5,]))
        self.transforms = Compose(tfs)

    def __getitem__(self, index):
        img = self.load_img(self.imgs, index, seed)

        return img

    def load_img(self, imgs_list, index, seed=None):
        img = Image.open(imgs_list[index]).convert('RGB')
        if(seed is not None):
            torch.manual_seed(seed)
            random.seed(seed)
        img = self.transforms(img)

        return img

    def __len__(self):
        return self.size