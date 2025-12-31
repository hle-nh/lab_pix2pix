# dataset_lab_subset.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.color import rgb2lab

class ColorizationSubsetLAB(Dataset):
    def __init__(self, root, img_size=256):
        self.files = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith((".jpg", ".png"))
        ]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.transform(img)

        lab = rgb2lab(img.permute(1, 2, 0).numpy())

        L  = torch.from_numpy(lab[:, :, 0]).unsqueeze(0) / 50.0 - 1.0
        ab = torch.from_numpy(lab[:, :, 1:]).permute(2, 0, 1) / 110.0

        return L.float(), ab.float()
