from model import *
from dataloader import *
import json
from PIL import Image


def test_pipeline(
    test_path,
    num_slices,
    test_save_path,
    path_best=None,
    roi=None,
    per_epoch=10,
    device="cpu",
    batch_size=1,
    mask_path=None
):
    X_test, mask_test = get_paths(test_path, num_slices, mask_path)
    if X_test[0][-4:] == '.png':
        img = np.array(Image.open(X_test[0])).astype(np.uint8)
    else:
        img = imread(X_test[0])
    sz = img.shape
    
        
    if len(mask_test) == 0:
        mask_test = None
    test_ds = ScrollDatasetTest(X_test, mask_test, roi=roi)
    test_batch_gen = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=12
    )
    print("Current device", device)
    model = PsiNet.load_from_checkpoint(
        path_best, train_save=None, val_save=None, per_epoch=per_epoch
    )
    model.read_num_slices(num_slices)
    model.eval()
    model.batch_size = batch_size
    model.test_save = test_save_path
    model.res = 0
    if not os.path.exists(test_save_path):
        os.mkdir(test_save_path)
    lx = 0
    ly = 0
    if roi[0] is None:
        if sz[0] % 32 != 0:
            lx = 32 - sz[0] % 32
        if sz[1] % 32 != 0:
            ly = 32 - sz[1] % 32
        
        model.test_data_type = "full"
    elif roi[0] == "vert_roi":
        model.res = sz[0]
        if sz[0] % 32 != 0:
            lx = 32 - sz[0] % 32
        if sz[1] % 32 != 0:
            ly = 32 - sz[1] % 32
        if (sz[0] + lx) * (sz[1] + ly) >= roi[1]:
            new_sz = roi[1] // (sz[0] + lx)
            if new_sz % 32 != 0:
                new_sz += 32 - new_sz % 32
            lx = (lx - (sz[1] + ly - new_sz) + 1) // 2
            
        model.test_data_type = "vert_roi"
        print(lx, ly)
    elif roi[0] == "hori_roi":
        model.res = sz[1]
        if sz[0] % 32 != 0:
            lx = 32 - sz[0] % 32
        if sz[1] % 32 != 0:
            ly = 32 - sz[1] % 32
        if (sz[0] + lx) * (sz[1] + ly) >= roi[1]:
            new_sz = roi[1] // (sz[1] + ly)
            if new_sz % 32 != 0:
                new_sz += 32 - new_sz % 32
            ly = (ly - (sz[0] + lx - new_sz) + 1) // 2
        print(lx, ly)
        model.test_data_type = "hori_roi"
    else:
        if sz[0] % 32 != 0:
            lx = 32 - sz[0] % 32
        if sz[1] % 32 != 0:
            ly = 32 - sz[1] % 32
        print(lx, ly)
        model.test_data_type = "patch"
    dictionary = {'lx': lx, 'ly': ly}
    logger = pl.loggers.TensorBoardLogger("", version=f"unet")
    if device == "cpu":
        trainer = pl.Trainer(logger=logger, accelerator="cpu", devices=1)
    else:
        trainer = pl.Trainer(logger=logger, accelerator="gpu", devices=-1)
    print(device)
    model.lx = lx
    model.ly = ly
    model.num_slices = num_slices
    trainer.test(model, dataloaders=test_batch_gen)
