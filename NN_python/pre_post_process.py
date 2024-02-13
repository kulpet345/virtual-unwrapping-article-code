import numpy as np
import torchvision.transforms as transforms
import copy


def postproc_align(bin_mask, lx, ly, tp, res):
    if lx >= 0 and ly >= 0:
        bin_mask = bin_mask[lx:, ly:]
    elif lx >= 0 and ly < 0:
        new_bin_mask = np.zeros((bin_mask.shape[0] - lx, bin_mask.shape[1] - ly)).astype(np.uint8)
        new_bin_mask[:, -ly:] = bin_mask[lx:]
        bin_mask = new_bin_mask
    elif lx < 0 and ly >= 0:
        new_bin_mask = np.zeros((bin_mask.shape[0] - lx, bin_mask.shape[1] - ly)).astype(np.uint8)
        new_bin_mask[-lx:, :] = bin_mask[:, ly:]
        bin_mask = new_bin_mask 
    else:
        new_bin_mask = np.zeros((bin_mask.shape[0] - lx, bin_mask.shape[1] - ly)).astype(np.uint8)
        new_bin_mask[-lx:, -ly:] = bin_mask
        bin_mask = new_bin_mask
    if tp == 'hori_roi':
        new_bin_mask = np.zeros((res, bin_mask.shape[1])).astype(np.uint8)
        new_bin_mask[:bin_mask.shape[0], :] = bin_mask
        bin_mask = new_bin_mask
    if tp == 'vert_roi':
        new_bin_mask = np.zeros((bin_mask.shape[0], res)).astype(np.uint8)
        new_bin_mask[:, :bin_mask.shape[1]] = bin_mask
        bin_mask = new_bin_mask
    return bin_mask 


def process_test_image(sample):
    sz1 = sample["image"].shape[0]
    sz2 = sample["image"].shape[1]
    if sample["mask"] is not None:
        sample["image"] = (sample["mask"] == 0) * (sample["image"])
    sample["image"] = transforms.ToTensor()(sample["image"]).float()

    pad1 = 0
    pad2 = 0
    if sz1 % 32 != 0:
        pad1 = 32 - sz1 % 32
    if sz2 % 32 != 0:
        pad2 = 32 - sz2 % 32
    sample["image"] = transforms.Pad((pad2, pad1, 0, 0))(sample["image"])
    sz1 += pad1
    sz2 += pad2
    mx_sz = 2**19 + 2**21
    if sample["roi"][1] is not None:
        mx_sz = sample["roi"][1]
    if sz1 * sz2 >= mx_sz:  # 2 ** 19 + 2 ** 21:#2 ** 21 + 2 ** 22:
        if sample["roi"][0] is None or sample["roi"][0] == "hori_roi":
            sz1 = mx_sz // sz2  # (2 ** 19 + 2 ** 21) // sz2#(2 ** 21 + 2 ** 22) // sz2
            if sz1 % 32 != 0:
                sz1 += 32 - sz1 % 32
        else:
            sz2 = mx_sz // sz1
            if sz2 % 32 != 0:
                sz2 += 32 - sz2 % 32
        sample["image"] = transforms.CenterCrop((sz1, sz2))(sample["image"])
    return sample["image"].float()


def process_test_image_patch(sample):
    roi = sample["roi"]
    sz1 = sample["image"].shape[0]
    sz2 = sample["image"].shape[1]
    pad1 = 0
    pad2 = 0
    if sample["mask"] is not None:
        sample["image"] = (sample["mask"] == 0) * (sample["image"])
    sample["image"] = transforms.ToTensor()(sample["image"]).float()
    if sz1 % 32 != 0:
        pad1 = 32 - sz1 % 32
    if sz2 % 32 != 0:
        pad2 = 32 - sz2 % 32
    sample["image"] = transforms.Pad((pad2, pad1, 0, 0))(sample["image"])
    sz1 += pad1
    sz2 += pad2

    mx_sz = 2**21 + 2**19
    if roi[1] is not None:
        mx_sz = roi[1]
    if sz1 * sz2 >= mx_sz:
        new_sz1 = int((mx_sz) ** 0.5)
        new_sz2 = new_sz1
        cnt1 = 0
        cnt2 = 0
        i = 0
        while i + new_sz1 < sz1:
            cnt1 += 1
            i = i + new_sz1 - 20
        i = 0
        while i + new_sz2 < sz2:
            cnt2 += 1
            i = i + new_sz2 - 20
        cnt1 += 1
        cnt2 += 1

        sample["new_img"] = [[0] * cnt2 for i in range(cnt1)]
        for i in range(cnt1):
            for j in range(cnt2):
                lx = i * (new_sz1 - 20)
                ly = j * (new_sz2 - 20)
                rx = lx + new_sz1
                ry = ly + new_sz2
                if rx > sample["image"].shape[1]:
                    rx = sample["image"].shape[1]
                if ry > sample["image"].shape[2]:
                    ry = sample["image"].shape[2]
                sample["new_img"][i][j] = sample["image"][:, lx:rx, ly:ry]
                pad1 = 0
                pad2 = 0
                if (rx - lx) % 32 != 0:
                    pad1 = 32 - (rx - lx) % 32
                if (ry - ly) % 32 != 0:
                    pad2 = 32 - (ry - ly) % 32
                sample["new_img"][i][j] = transforms.Pad((pad2, pad1, 0, 0))(
                    sample["new_img"][i][j]
                )
        sample["image"] = None
        sample["sz"] = (copy.deepcopy(sz1), copy.deepcopy(sz2))
        return sample["new_img"], sample["sz"]

    sample["new_img"] = [[sample["image"].float()]]
    sample["image"] = None
    sample["sz"] = (copy.deepcopy(sz1), copy.deepcopy(sz2))
    return sample["new_img"], sample["sz"]


def postproc_test_img(sample):
    if len(sample["pred_dist"]) == 1 and len(sample["pred_dist"][0]) == 1:
        return sample["pred_dist"][0][0].cpu().numpy()[0]
    new_sz1 = int((2**21 + 2**19) ** 0.5)
    if sample["roi"][1] is not None:
        new_sz1 = int((sample["roi"][1]) ** 0.5)
    new_sz2 = new_sz1

    sz1 = copy.deepcopy(sample["sz"][0])
    sz2 = copy.deepcopy(sample["sz"][1])
    cnt1 = 0
    cnt2 = 0
    i = 0
    while i + new_sz1 < sz1:
        cnt1 += 1
        i = i + new_sz1 - 20
    i = 0
    while i + new_sz2 < sz2:
        cnt2 += 1
        i = i + new_sz2 - 20
    cnt1 += 1
    cnt2 += 1
    sample["new_pred_dist"] = np.zeros(
        (sample["pred_dist"][0][0].shape[1], sz1, sz2)
    ).astype(np.float32)
    for i in range(cnt1):
        for j in range(cnt2):
            lx = i * (new_sz1 - 20)
            ly = j * (new_sz2 - 20)
            rx = lx + new_sz1
            ry = ly + new_sz2
            if rx > sz1:
                rx = sz1
            if ry > sz2:
                ry = sz2
            pad1 = 0
            pad2 = 0
            if (rx - lx) % 32 != 0:
                pad1 = 32 - (rx - lx) % 32
            if (ry - ly) % 32 != 0:
                pad2 = 32 - (ry - ly) % 32
            lenx = rx - lx
            leny = ry - ly
            prevlx = copy.deepcopy(lx)
            prevly = copy.deepcopy(ly)
            prevrx = copy.deepcopy(rx)
            prevry = copy.deepcopy(ry)
            if i != 0:
                lx += 10
            if j != 0:
                ly += 10
            if i != cnt1 - 1:
                rx -= 10
            if j != cnt2 - 1:
                ry -= 10
            sample["new_pred_dist"][:, lx:rx, ly:ry] = sample["pred_dist"][i][j][
                :,
                :,
                pad1
                + (lx - prevlx) : sample["pred_dist"][i][j][0][0].shape[0]
                - (prevrx - rx),
                pad2
                + (ly - prevly) : sample["pred_dist"][i][j][0][0].shape[1]
                - (prevry - ry),
            ]
    return sample["new_pred_dist"]
 

