from torch.utils.data import Dataset
import os
import linecache
from PIL import Image
import numpy as np
import torch
from tool.common import resizeNormalize


class customDataset(Dataset):
    def __init__(self, dataset_path,gt_filename):
        super(customDataset, self).__init__()
        with open(os.path.join(dataset_path,gt_filename),encoding='utf-8') as f:
            lines=f.readlines()
        self.length=len(lines)
        self.gt_filename=gt_filename
        self.dataset_path=dataset_path
        

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        line=linecache.getline(os.path.join(self.dataset_path,self.gt_filename),index+1)
        index=line.find('\t')
        label=line[index+1:].strip('\n')
        image_path=os.path.join(self.dataset_path, 'images', line[:index])
        img = Image.open(image_path).convert('RGB')
        return (img,label)



class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            mean_ratio = np.mean(ratios)
            imgW = int(np.floor(mean_ratio * imgH))
            imgW = max(self.imgW, imgW)
            # ratios.sort()
            # max_ratio = ratios[-1]
            # imgW = int(np.floor(max_ratio * imgH))
            # imgW = max(self.imgW, imgW)  # assure imgH >= imgW

        images = [resizeNormalize((imgH, imgW),image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        # unloader = transforms.ToPILImage()
        # iii = unloader(images[0][0])
        # iii.save('.aaa.JPG')
    
        return images, labels

