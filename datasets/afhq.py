import os
from PIL import Image
import torch
from torch.utils.data import Dataset 

class AFHQ(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.subdirs = os.listdir(root)
        self.img_paths = []
        
        for subdir in self.subdirs:
            subdir_path = os.path.join(root, subdir)
            img_paths = os.listdir(subdir_path)
            
            self.img_paths += [os.path.join(subdir_path, img_path) for img_path in img_paths]
            
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        target = [0]
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        # Transform
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target