from test_pipeline import test_pipeline

import json
import sys
import os
from PIL import Image
from tifffile import imread
import numpy as np


def get_roi_size(mb_size, path_slices):
    
    ends = set([".tif", ".tiff", ".png"])
    for f in os.listdir(path_slices):
        if os.path.splitext(f)[1] in ends:
            if os.path.splitext(f)[1] == ".png":
                img = np.array(Image.open(os.path.join(path_slices, f)))
            else:
                img = imread(os.path.join(path_slices, f))
            sz = img.shape[0] * img.shape[1]
            #11000 - etalon Megabytes, 6291456 - etalon patch-size
            patch_sz = int(round(6291456 / 11000 * mb_size))
            if patch_sz % 32 != 0:
                patch_sz += 32 - patch_sz % 32
            return patch_sz

if __name__ == '__main__':
    config_path = sys.argv[1]

    with open(config_path) as f:
        CFG = json.load(f)
    
    os.chdir(os.path.dirname(config_path))
        
    if not os.path.exists(CFG["FOLDER_PATH_MASK"]):
        os.mkdir(CFG["FOLDER_PATH_MASK"])
    if not os.path.exists(CFG["FOLDER_PATH_SLICES"]):
        raise OSError("FOLDER_PATH_SLICES dir is not found")
    if not os.path.exists(CFG["FILE_PATH_NN_WEIGHTS"]):
        raise OSError("Bad weights path")
    
    
    roi_size = get_roi_size(CFG["NN_INFERENCE_MEMORY_Mb"], CFG["FOLDER_PATH_SLICES"])
    #if "FOLDER_PATH_SEG_SLICES" in CFG:
    #    CFG["FOLDER_PATH_SLICES"] = CFG["FOLDER_PATH_SEG_SLICES"]
    mask_path = None
    if "FOLDER_PATH_NOISE_MASK" in CFG:
        mask_path = CFG["FOLDER_PATH_NOISE_MASK"]

        
    test_pipeline(test_path=CFG["FOLDER_PATH_SLICES"],
              num_slices=CFG["FILE_PATH_NUM_PATH"],
              roi=("patch", roi_size),
              path_best=CFG["FILE_PATH_NN_WEIGHTS"],
              test_save_path=CFG["FOLDER_PATH_MASK"],
              device=CFG["NN_INFERENCE_DEVICE"],
              mask_path=mask_path)
    name = os.path.basename(config_path)
    os.system("cd unfolder/build && ./Unfolder ../../" + name)

