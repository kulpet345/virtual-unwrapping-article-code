from torch.utils.data import Dataset
import os
import numpy as np
import cv2
from pre_post_process import process_test_image, process_test_image_patch
import torch
from tifffile import imread


def get_paths(path, num_slices, mask_path=None):
    _, _, filenames = next(os.walk(path))
    mask_filenames = []
    if mask_path is not None:
        _, _, mask_filenames = next(os.walk(mask_path))

    images_paths = []
    mask_paths = []
    nums = set()
    with open(num_slices, "r") as f:
        lines = f.readlines()
        for line in lines:
            nums.add(int(line))
    for filename in sorted(filenames):
        if os.path.splitext(filename)[1] == ".json":
            continue
        if int(filename[-8:-4]) in nums: 
            images_paths.append(os.path.join(path, filename))
    for filename in mask_filenames:
        if int(filename[4:8]) in nums: 
            mask_paths.append(os.path.join(mask_path, filename))

    res, image_files = check_consistence(images_paths)
    res1, mask_files = check_consistence(mask_paths)

    if not res or not res1:
        raise OSError("Consistent error, exit")
    stacked_mask = []
    if len(mask_files) != 0:
        stacked_mask = np.stack(mask_files)
    return np.stack(image_files), stacked_mask
    
    
def check_consistence(images_paths):
    paths = []
    for path in images_paths:
        if os.path.splitext(path)[1] == ".json":
            continue
        corr_names = set([".tif", ".tiff", ".png"])
        if os.path.splitext(path)[1] not in corr_names:
            print("Inconsistent file {}".format(path))
            return False, []
        paths.append(path)
    return True, paths


class ScrollDatasetTest(Dataset):
    def __init__(self, images, masks, roi=None):
        self.images = sorted(images)
        self.masks = None
        if masks is not None:
            self.masks = sorted(masks)
        self.roi = roi

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im_name = self.images[idx]
        mask_name = None
        if self.masks is not None:
            mask_name = self.masks[idx]
        pos = -1
        for i in range(len(im_name) - 1, -1, -1):
            if im_name[i] == ".":
                pos = i
                break
        if im_name[pos:] == ".png":
            image = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)
            ret2, th2 = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif im_name[pos:] == ".tif" or im_name[pos:] == ".tiff":
            img = imread(im_name, key=0)
            image = np.round((img - img.min()) / (img.max() - img.min()) * 255).astype(
                np.uint8
            )
            ret2, th2 = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        mask = None
        if mask_name is not None:
            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

        q = 0.995
        cnt = np.sum(th2 > 0)
        val = np.quantile(image, 1 - (1 - q) * cnt / (image.shape[0] * image.shape[1]))
        image = np.clip(image, 0, val) / val

        sample = {"image": image, "roi": self.roi, "clip_val": val, "mask": mask}

        if (
            self.roi[0] is None
            or self.roi[0] == "hori_roi"
            or self.roi[0] == "vert_roi"
        ):
            sample["roi"] = self.roi
            image = process_test_image(sample)
        elif self.roi[0] == "patch":
            sample["roi"] = (None, self.roi[1])
            images, sz = process_test_image_patch(sample)
        if (
            self.roi[0] is None
            or self.roi[0] == "hori_roi"
            or self.roi[0] == "vert_roi"
        ):
            return torch.FloatTensor(image), sample["clip_val"]
        else:
            return images, sz, sample["clip_val"], self.roi

