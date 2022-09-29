import glob
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

class ImageFolder(Dataset):
    def __init__(
        self,
        dataset,
        data_dir,
        transform
    ):
        super(ImageFolder, self).__init__()
        if dataset == 'ImageNet':
            self.fnames = list(glob.glob(data_dir + '/train/*/*.JPEG'))
        elif dataset == 'COCO':
            self.fnames = list(glob.glob(data_dir + '/train2017/*.jpg'))
        elif dataset == 'COCOplus':
            self.fnames = list(glob.glob(data_dir + '/train2017/*.jpg')) + list(glob.glob(data_dir + '/unlabeled2017/*.jpg'))
        elif dataset == 'COCOval':
            self.fnames = list(glob.glob(data_dir + '/val2017/*.jpg'))
        else:
            raise NotImplementedError

        self.fnames = np.array(self.fnames) # to avoid memory leak
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fpath = self.fnames[idx]
        image = Image.open(fpath).convert('RGB')
        return self.transform(image)
