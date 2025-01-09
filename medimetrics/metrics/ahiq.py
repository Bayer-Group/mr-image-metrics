from typing import Any, List

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.resnet import BasicBlock, Bottleneck
from timm.models.vision_transformer import Block
from torchvision.ops.deform_conv import DeformConv2d

from medimetrics.base import FullRefMetric


class deform_fusion(nn.Module):
    def __init__(
        self, patch_size: int, in_channels: int = 768 * 5, cnn_channels: int = 256 * 3, out_channels: int = 256 * 3
    ) -> None:
        super().__init__()
        # in_channels, out_channels, kernel_size, stride, padding
        self.d_hidn = 512
        if patch_size == 8:
            stride = 1
        else:
            stride = 2
        self.conv_offset = nn.Conv2d(in_channels, 2 * 3 * 3, 3, 1, 1)
        self.deform = DeformConv2d(cnn_channels, out_channels, 3, 1, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=self.d_hidn, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.d_hidn, out_channels=out_channels, kernel_size=3, padding=1, stride=stride),
        )

    def forward(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor) -> torch.Tensor:
        vit_feat = F.interpolate(vit_feat, size=cnn_feat.shape[-2:], mode="nearest")
        offset = self.conv_offset(vit_feat)
        deform_feat = self.deform(cnn_feat, offset)
        deform_feat = self.conv1(deform_feat)

        return deform_feat


class Pixel_Prediction(nn.Module):
    def __init__(self, inchannels: int = 768 * 5 + 256 * 3, outchannels: int = 256, d_hidn: int = 1024):
        super().__init__()
        self.d_hidn = d_hidn
        self.down_channel = nn.Conv2d(inchannels, outchannels, kernel_size=1)
        self.feat_smoothing = nn.Sequential(
            nn.Conv2d(in_channels=256 * 3, out_channels=self.d_hidn, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.d_hidn, out_channels=512, kernel_size=3, padding=1),
        )

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1), nn.ReLU())
        self.conv_attent = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1), nn.Sigmoid())
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
        )

    def forward(
        self, f_dis: torch.Tensor, f_ref: torch.Tensor, cnn_dis: torch.Tensor, cnn_ref: torch.Tensor
    ) -> torch.Tensor:
        f_dis = torch.cat((f_dis, cnn_dis), 1)
        f_ref = torch.cat((f_ref, cnn_ref), 1)
        f_dis = self.down_channel(f_dis)
        f_ref = self.down_channel(f_ref)

        f_cat = torch.cat((f_dis - f_ref, f_dis, f_ref), 1)

        feat_fused = self.feat_smoothing(f_cat)
        feat = self.conv1(feat_fused)
        f = self.conv(feat)
        w = self.conv_attent(feat)
        pred = (f * w).sum(dim=2).sum(dim=2) / w.sum(dim=2).sum(dim=2)

        return pred


def get_resnet_feature(save_output: Any) -> torch.Tensor:
    feat = torch.cat((save_output.outputs[0], save_output.outputs[1], save_output.outputs[2]), dim=1)
    return feat


def get_vit_feature(outputs: List[torch.Tensor]) -> torch.Tensor:
    feat = torch.cat(
        (
            outputs[0][:, 1:, :],
            outputs[1][:, 1:, :],
            outputs[2][:, 1:, :],
            outputs[3][:, 1:, :],
            outputs[4][:, 1:, :],
        ),
        dim=2,
    )
    return feat


def five_point_crop(idx: int, d_img: torch.Tensor, r_img: torch.Tensor, crop_size: int = 224) -> torch.Tensor:
    new_h = crop_size
    new_w = crop_size
    if len(d_img.shape) == 3:
        c, h, w = d_img.shape
    else:
        b, c, h, w = d_img.shape
    center_h = h // 2
    center_w = w // 2
    if idx == 0:
        top = 0
        left = 0
    elif idx == 1:
        top = 0
        left = w - new_w
    elif idx == 2:
        top = h - new_h
        left = 0
    elif idx == 3:
        top = h - new_h
        left = w - new_w
    elif idx == 4:
        top = center_h - new_h // 2
        left = center_w - new_w // 2
    elif idx == 5:
        left = 0
        top = center_h - new_h // 2
    elif idx == 6:
        left = w - new_w
        top = center_h - new_h // 2
    elif idx == 7:
        top = 0
        left = center_w - new_w // 2
    elif idx == 8:
        top = h - new_h
        left = center_w - new_w // 2
    if len(d_img.shape) == 3:
        d_img_org = d_img[:, top : top + new_h, left : left + new_w]
        r_img_org = r_img[:, top : top + new_h, left : left + new_w]
    else:
        d_img_org = d_img[:, :, top : top + new_h, left : left + new_w]
        r_img_org = r_img[:, :, top : top + new_h, left : left + new_w]
    return d_img_org, r_img_org


class SaveOutput:
    def __init__(self) -> None:
        self.outputs: List[torch.Tensor] = []

    def __call__(self, module: nn.Module, module_in: nn.Module, module_out: nn.Module) -> None:
        self.outputs.append(module_out)

    def clear(self) -> None:
        self.outputs = []


class AHIQ(FullRefMetric):
    def __init__(self, patch_size: int = 8, crop_size: int = 224, n_ensemble: int = 9) -> None:

        # use ensemble = 20 (> 9) for random patch cropping

        # default options
        self.patch_size = patch_size
        self.crop_size = crop_size
        self.n_ensemble = n_ensemble

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.resnet50 = timm.create_model("resnet50", pretrained=True).to(self.device)
        if self.patch_size == 8:
            self.vit = timm.create_model("vit_base_patch8_224", pretrained=True).to(self.device)
        else:
            self.vit = timm.create_model("vit_base_patch16_224", pretrained=True).to(self.device)
        self.deform_net = deform_fusion(patch_size=self.patch_size).to(self.device)
        self.regressor = Pixel_Prediction().to(self.device)

        self.save_output = SaveOutput()

        hook_handles = []
        for layer in self.resnet50.modules():
            if isinstance(layer, Bottleneck):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

    def compute(self, image_true: np.ndarray, image_test: np.ndarray, **kwargs: Any) -> float:
        """
        Parameters:
        -----------
        image_true: np.array (H, W)
            Reference image
        image_test: np.array (H, W)
            Image to be evaluated against the reference image
        """

        image_true_T = torch.Tensor(image_true.copy()).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
        image_test_T = torch.Tensor(image_test.copy()).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

        with torch.no_grad():
            # for data in tqdm(self.test_loader):
            # d_img_org = data['d_img_org'].cuda()
            # r_img_org = data['r_img_org'].cuda()
            d_img_org = image_test_T.to(self.device)
            r_img_org = image_true_T.to(self.device)
            # d_img_name = data['d_img_name']
            pred = 0.0
            for i in range(self.n_ensemble):
                b, c, h, w = r_img_org.size()
                if self.n_ensemble > 9:
                    # config.crop_size
                    new_h = 224
                    new_w = 224
                    top = np.random.randint(0, h - new_h + 1)
                    left = np.random.randint(0, w - new_w + 1)
                    r_img = r_img_org[:, :, top : top + new_h, left : left + new_w]
                    d_img = d_img_org[:, :, top : top + new_h, left : left + new_w]
                elif self.n_ensemble == 1:
                    r_img = r_img_org
                    d_img = d_img_org
                else:
                    d_img, r_img = five_point_crop(i, d_img=d_img_org, r_img=r_img_org, crop_size=self.crop_size)
                d_img = d_img.to(self.device)
                r_img = r_img.to(self.device)
                _x = self.vit(d_img)
                vit_dis = get_vit_feature(self.save_output.outputs)
                self.save_output.outputs.clear()

                _y = self.vit(r_img)
                vit_ref = get_vit_feature(self.save_output.outputs)
                self.save_output.outputs.clear()
                B, N, C = vit_ref.shape
                if self.patch_size == 8:
                    H, W = 28, 28
                else:
                    H, W = 14, 14
                assert H * W == N
                vit_ref = vit_ref.transpose(1, 2).view(B, C, H, W)
                vit_dis = vit_dis.transpose(1, 2).view(B, C, H, W)

                _ = self.resnet50(d_img)
                cnn_dis = get_resnet_feature(self.save_output)
                self.save_output.outputs.clear()
                cnn_dis = self.deform_net(cnn_dis, vit_ref)

                _ = self.resnet50(r_img)
                cnn_ref = get_resnet_feature(self.save_output)
                self.save_output.outputs.clear()
                cnn_ref = self.deform_net(cnn_ref, vit_ref)
                pred += self.regressor(vit_dis, vit_ref, cnn_dis, cnn_ref)

            # end for loop
            pred /= self.n_ensemble
            # for i in range(len(d_img_name)):
            #    line = "%s,%f\n" % (d_img_name[i], float(pred.squeeze()[i]))
            #    f.write(line)

        # no squeeze needed?
        return pred.cpu().numpy().squeeze()


if __name__ == "__main__":
    from medimetrics.distortions import GaussianBlur

    # create test images:
    image_true = np.random.rand(240, 240)
    ahiq = AHIQ()
    blur = GaussianBlur(10)
    for s in range(10):
        image_test = blur(image_true, s)

        print(s, ": ", ahiq.compute(image_true, image_test))
