from torch import nn
import pytorch_lightning as pl
import torch
import os
import numpy as np
import cv2
from pre_post_process import postproc_test_img, postproc_align
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=False):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class UNetModule(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class PsiNet(pl.LightningModule):
    """
    Adapted from Vanilla UNet implementation - https://github.com/lopuhin/mapillary-vistas-2017/blob/master/unet_models.py
    """

    output_downscaled = 1
    module = UNetModule

    def __init__(
        self,
        train_save,
        val_save,
        per_epoch,
        input_channels: int = 1,
        filters_base: int = 32,
        down_filter_factors=(1, 2, 4, 8, 16),
        up_filter_factors=(1, 2, 4, 8, 16),
        bottom_s=4,
        num_classes=2,
        add_output=True,
    ):
        super().__init__()
        self.train_save = train_save
        self.val_save = val_save
        self.per_epoch = per_epoch
        self.num_classes = num_classes
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        self.down.append(self.module(input_channels, down_filter_sizes[0]))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.down.append(self.module(down_filter_sizes[prev_i], nf))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.up.append(
                self.module(down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i])
            )
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample1 = nn.Upsample(scale_factor=2)
        upsample_bottom1 = nn.Upsample(scale_factor=bottom_s)
        upsample2 = nn.Upsample(scale_factor=2)
        upsample_bottom2 = nn.Upsample(scale_factor=bottom_s)
        upsample3 = nn.Upsample(scale_factor=2)
        upsample_bottom3 = nn.Upsample(scale_factor=bottom_s)

        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers1 = [upsample1] * len(self.up)
        self.upsamplers1[-1] = upsample_bottom1
        self.upsamplers2 = [upsample2] * len(self.up)
        self.upsamplers2[-1] = upsample_bottom2
        self.upsamplers3 = [upsample3] * len(self.up)
        self.upsamplers3[-1] = upsample_bottom3

        self.add_output = add_output
        if add_output:
            self.conv_final1 = nn.Conv2d(up_filter_sizes[0], num_classes, 1)
        if add_output:
            self.conv_final2 = nn.Conv2d(up_filter_sizes[0], num_classes, 1)
        if add_output:
            self.conv_final3 = nn.Conv2d(up_filter_sizes[0], 1, 1)
            
    def read_num_slices(self, num_slices):
        self.nums = []
        with open(num_slices, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.nums.append(int(line))
        

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        x_out = xs[-1]
        x_out1 = x_out
        x_out2 = x_out
        x_out3 = x_out

        # Decoder mask segmentation
        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers1, self.up))
        ):
            x_out1 = upsample(x_out1)
            x_out1 = up(torch.cat([x_out1, x_skip], 1))

        # Decoder contour estimation
        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers2, self.up))
        ):
            x_out2 = upsample(x_out2)
            x_out2 = up(torch.cat([x_out2, x_skip], 1))

        # Regression
        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers3, self.up))
        ):
            x_out3 = upsample(x_out3)
            x_out3 = up(torch.cat([x_out3, x_skip], 1))

        if self.add_output:
            x_out1 = self.conv_final1(x_out1)
            if self.num_classes > 1:
                x_out1 = F.log_softmax(x_out1, dim=1)

        if self.add_output:
            x_out2 = self.conv_final2(x_out2)
            if self.num_classes > 1:
                x_out2 = F.log_softmax(x_out2, dim=1)

        if self.add_output:
            x_out3 = self.conv_final3(x_out3)
            x_out3 = F.sigmoid(x_out3)

        return [x_out1, x_out2, x_out3]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return [optimizer], []

    def test_step(self, batch, batch_idx):
        if self.test_data_type == "patch":
            raw, sz, clip_val, roi = batch
            pred_mask_list = [[0] * len(raw[0]) for i in range(len(raw))]
            pred_boundary_list = [[0] * len(raw[0]) for i in range(len(raw))]
            pred_dist_list = [[0] * len(raw[0]) for i in range(len(raw))]
            for i in range(len(raw)):
                for j in range(len(raw[0])):
                    pred_mask_patch, pred_boundary_patch, pred_dist_patch = self(
                        raw[i][j]
                    )
                    pred_mask_list[i][j] = pred_mask_patch
                    pred_boundary_list[i][j] = pred_boundary_patch
                    pred_dist_list[i][j] = pred_dist_patch.cpu()
                    pred_dist_list[i][j] = pred_dist_list[i][j].cpu()
                    raw[i][j] = raw[i][j].cpu()

            sample = {"pred_dist": pred_dist_list, "sz": sz, "roi": roi}
            pred_dist = postproc_test_img(sample)
            pred_dist_cpu = pred_dist[0] * 16
            bin_mask = np.array((pred_dist_cpu >= 3) * 255).astype(np.uint8)
            bin_mask = postproc_align(bin_mask, self.ly, self.lx, self.test_data_type, self.res)
            cv2.imwrite(
                os.path.join(self.test_save, "bin_mask"
                + str(self.nums[batch_idx * self.batch_size]).zfill(4)
                + ".png"),
                bin_mask
            )
        else:
            raw, clip_val = batch
            pred_mask, pred_boundary, pred_dist = self(raw)
            if not os.path.exists(self.test_save):
                os.mkdir(self.test_save)
            for i in range(1):
                pred_dist_cpu = pred_dist[i][0].cpu().numpy() * 16
                bin_mask = np.array((pred_dist_cpu >= 3) * 255).astype(np.uint8)
                bin_mask = postproc_align(bin_mask, self.ly, self.lx, self.test_data_type, self.res)
                cv2.imwrite(
                    os.path.join(self.test_save, "bin_mask"
                    + str(self.nums[batch_idx * self.batch_size + i]).zfill(4)
                    + ".png"),
                    bin_mask
                )
                color_DT = np.zeros((pred_dist_cpu.shape[0], pred_dist_cpu.shape[1], 3)).astype(np.uint8)
                color_DT += (pred_dist_cpu >= 0.5) * (pred_dist_cpu < 1.5) * [255, 0, 0]
                color_DT += (pred_dist_cpu >= 1.5) * (pred_dist_cpu < 2.5) * [255, 60, 0]
                color_DT += (pred_dist_cpu >= 2.5) * (pred_dist_cpu < 3.5) * [255, 120, 0]
                color_DT += (pred_dist_cpu >= 3.5) * (pred_dist_cpu < 4.5) * [255, 180, 0]
                color_DT += (pred_dist_cpu >= 4.5) * [255, 255, 0]
                cv2.imwrite(
                    os.path.join(self.test_save, "DT"
                    + str(self.nums[batch_idx * self.batch_size + i]).zfill(4)
                    + ".png"),
                    color_DT
                )


    def image_prediction(self, image):
        '''
        Get image prediction, png-numpy image
        '''
        ret2, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        q = 0.995
        cnt = np.sum(th2 > 0)
        val = np.quantile(image, 1 - (1 - q) * cnt / (image.shape[0] * image.shape[1]))
        image = np.clip(image, 0, val) / val
        sz1 = image.shape[0]
        sz2 = image.shape[1]
        pad1 = 0
        pad2 = 0
        if sz1 % 32 != 0:
            pad1 = 32 - sz1 % 32
        if sz2 % 32 != 0:
            pad2 = 32 - sz2 % 32
        tensor_image = transforms.Pad((pad2, pad1, 0, 0))(
            transforms.ToTensor()(image).float()
        )
        raw_tensor = tensor_image.float().unsqueeze(0)
        pred_mask, pred_boundary, pred_dist = self(raw_tensor)
        return pred_mask, pred_boundary, pred_dist

    def tensor_image_prediction(self, raw_tensor):
        '''
        Get image prediction from tensor
        '''
        pred_mask, pred_boundary, pred_dist = self(raw_tensor)
        return pred_mask, pred_boundary, pred_dist

