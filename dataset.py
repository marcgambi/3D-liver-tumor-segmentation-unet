import os
import nibabel as nib
import torch
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    """
    Custom PyTorch Dataset for loading NIfTI medical images and segmentation labels.
    """

    def __init__(self, annotations_file, img_dir, label_dir, normalize=False):
        self.img_labels = annotations_file
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.normalize = normalize

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_name = self.img_labels.loc[idx, "image_path"]
        label_name = self.img_labels.loc[idx, "label_path"]

        image_path = os.path.join(self.img_dir, image_name)
        label_path = os.path.join(self.label_dir, label_name)

        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        if self.normalize:
            image = image / image.max()

        return image, label
