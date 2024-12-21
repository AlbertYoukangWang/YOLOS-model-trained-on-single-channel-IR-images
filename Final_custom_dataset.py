# Last Update: 20/12/2024
# Dataset Definitions for fetching our data

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class FinalCustomDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        
        self.image_files = os.listdir(image_dir)
        self.label_files = os.listdir(label_dir)

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, i):
        image_path = os.path.join(self.image_dir, self.image_files[i])
        image_PIL = Image.open(image_path).convert('L') # Ensure it's 1-dim, sometimes loading/saving turns images from 1-dim to 3-dim.
        image_tensor = transforms.ToTensor()(image_PIL)
        # print(f"***Image tensor shape: {image_tensor.shape}")
        image = image_tensor

        label_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_filename)
        label_read = np.loadtxt(label_path)
        label = torch.tensor(label_read)
        
        # Handle the case where there's only 1 object in image
        if label.dim() == 1:
            label = label.unsqueeze(0) # Have to assign back!

        # Change from (xmax, ymax, xmin, ymin) -> (xmin, ymin, xmax, ymax)
        boxes = label[:, 1:]
        swapped_boxes = torch.cat([boxes[:, 2:4], boxes[:, 0:2]], dim=1)

        target = {'class_labels': label[:, 0].long(), 'boxes': swapped_boxes}

        return image, target
    
if __name__ == "__main__":
    dataset = FinalCustomDataset("/mnt/extra-dtc/Infrared-Object-Detection/datasets/infrared/images",
                                  "/mnt/extra-dtc/Infrared-Object-Detection/datasets/infrared/labels")
    for image, target in dataset:
        print(image)
        print(target)
        break
