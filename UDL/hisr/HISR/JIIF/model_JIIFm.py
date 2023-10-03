from torch import optim
import torch
from edsr import make_edsr_baseline, make_coord
import torch.nn.functional as F
from UDL.pansharpening.common.evaluate import analysis_accu
from UDL.Basis.criterion_metrics import *
from UDL.Basis.pytorch_msssim.cal_ssim import SSIM


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class JIIF_conv_mean(nn.Module):
    def __init__(self, feat_dim=128, guide_dim=128, spa_edsr_num=4, spe_edsr_num=4, mlp_dim=[256, 128], NIR_dim=33, lr_hw=(16, 16)):
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.mlp_dim = mlp_dim
        self.NIR_dim = NIR_dim

        self.spatial_encoder = make_edsr_baseline(n_resblocks=spa_edsr_num, n_feats=self.guide_dim, n_colors=34)
        self.spectral_encoder = make_edsr_baseline(n_resblocks=spe_edsr_num, n_feats=self.feat_dim, n_colors=31)
        # 添加新encoder 2023 3 22
        self.spectral_encoder_34 = make_edsr_baseline(n_resblocks=spe_edsr_num, n_feats=self.feat_dim, n_colors=34)
        self.spectral_encoder_127 = make_edsr_baseline(n_resblocks=spe_edsr_num, n_feats=self.feat_dim, n_colors=127)
        # 改成可修改lr大小
        self.spatial_encoder_lr = nn.AdaptiveMaxPool2d(lr_hw)

        imnet_in_dim = self.feat_dim*2 + self.guide_dim + 2

        self.imnet = MLP(imnet_in_dim, out_dim=NIR_dim, hidden_list=self.mlp_dim)
        self.decoder = nn.Sequential(nn.Conv2d(NIR_dim-1, NIR_dim-1, kernel_size=3, padding=1, bias=False),
                                    nn.Conv2d(NIR_dim-1, 31, kernel_size=5, padding=2, bias=False))
        # I know that. One of the values is depth, and another is the weight.

    def query(self, feat, coord, hr_guide, lr_guide):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

        b, c, h, w = feat.shape  # lr  7x128x16x16
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        feat = torch.cat([lr_guide, feat], dim=1)
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)

        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]  no influence

        rx = 1 / h
        ry = 1 / w

        preds = []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]

                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]


                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1)
                pred = self.imnet(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # [B, N, 2, kk]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1)
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)

        return ret

    def forward(self, HR_MSI, lms, LR_HSI):
        # HR_MSI Bx3x64x64
        # lms Bx31x64x64
        # LR_HSI Bx31x16x16

        _, _, H, W = HR_MSI.shape
        coord = make_coord([H, W]).to(HR_MSI.device)
        feat = torch.cat([HR_MSI, lms], dim=1)
        hr_spa = self.spatial_encoder(feat)  # Bx128xHxW
        lr_spa = self.spatial_encoder_lr(hr_spa)
        if LR_HSI.shape[1] == 34:
            lr_spe = self.spectral_encoder_34(LR_HSI)  # Bx128xhxw The feature map of LR-HSI
        elif LR_HSI.shape[1] == 127:
            lr_spe = self.spectral_encoder_127(LR_HSI)
        else:
            raise ValueError('LR channel num is wrong')

        NIR_feature = self.query(lr_spe, coord, hr_spa, lr_spa)  # BxCxHxW

        # next, we can use the Transformer as the decoder
        output = self.decoder(NIR_feature)

        output = lms + output

        return output


def pixel_shuffle_inv(tensor, scale_factor):
    B, C, H, W = tensor.shape
    if H % scale_factor != 0 or W % scale_factor != 0:
        raise ValueError('height and widht of tensor must be divisible by '
                         'scale_factor.')

    c = C * (scale_factor * scale_factor)
    h = H // scale_factor
    w = W // scale_factor

    tensor = tensor.reshape(
        [B, C, h, scale_factor, w, scale_factor])
    # new axis: [num, ch, scale_factor, scale_factor, new_height, new_width]
    tensor = tensor.permute([0, 1, 2, 4, 3, 5])
    tensor = tensor.reshape([B, c, h, w])
    return tensor


class JIIF_multiple2(nn.Module):
    def __init__(self, h, w, m_scale):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((h, w))
        self.pool2 = nn.AdaptiveAvgPool2d((h//m_scale, w//m_scale))
        self.model_1 = JIIF_conv_mean(64, 64, lr_hw=(h, w)).cuda()
        self.model_2 = JIIF_conv_mean(64, 64, lr_hw=(h//m_scale, w//m_scale)).cuda()
        self.out_conv = torch.nn.Conv2d(62, 31, 1).to('cuda')

    def forward(self, HR_MSI, lms, LR_HSI):
        # 第一种规模：下采样HR，与LR cat到一起，过JIIF
        HR_MSI_L_1 = self.pool1(HR_MSI)

        LR_MSI_H_1 = torch.cat([LR_HSI, HR_MSI_L_1], dim=1)         # B*31*h*w

        output_1 = self.model_1(HR_MSI, lms, LR_MSI_H_1)            # B*31*H*W

        # 第二种规模：LR pixel_shuffle_inv，下采样HR，与新LR cat到一起，过JIIF
        LR_HSI_ps_2 = pixel_shuffle_inv(LR_HSI, scale_factor=2)     # B* 124 * h/2 * w/2

        HR_MSI_L_2 = self.pool2(HR_MSI)
        LR_MSI_H_2 = torch.cat([LR_HSI_ps_2, HR_MSI_L_2], dim=1)

        output_2 = self.model_2(HR_MSI, lms, LR_MSI_H_2)         # B*31*H*W

        # 合并多种结果
        output = torch.cat([output_1, output_2], dim=1)

        output = self.out_conv(output)

        return output

    def train_step(self, batch, *args, **kwargs):
        gt, up, hsi, msi = batch['gt'].cuda(), \
                           batch['up'].cuda(), \
                           batch['lrhsi'].cuda(), \
                           batch['rgb'].cuda()
        sr = self(msi, up, hsi)
        loss = self.criterion(sr, gt, *args, **kwargs)
        log_vars = {}
        with torch.no_grad():
            metrics = analysis_accu(gt, sr, 4, choices=4)
            log_vars.update(metrics)

        return {'loss': loss, 'log_vars': log_vars}

    def eval_step(self, batch, *args, **kwargs):
        gt, up, hsi, msi = batch['gt'].cuda(), \
                           batch['up'].cuda(), \
                           batch['lrhsi'].cuda(), \
                           batch['rgb'].cuda()

        sr1 = self.forward(msi, up, hsi)

        with torch.no_grad():
            metrics = analysis_accu(gt[0].permute(1, 2, 0), sr1[0].permute(1, 2, 0), 4)
            metrics.update(metrics)

        return sr1, metrics

    def set_metrics(self, criterion, rgb_range=1.0):
        self.rgb_range = rgb_range
        self.criterion = criterion


def build(args):
    scheduler = None
    mode = "one"
    loss1 = nn.L1Loss().cuda()
    # weight_dict = {'Loss': 1}
    # losses = {'Loss': loss1}

    g_ssim = SSIM(size_average=True)
    loss2 = g_ssim.cuda()
    weight_dict = {'Loss': 1, 'ssim_loss': 0.1}
    losses = {'Loss': loss1, 'ssim_loss': loss2}  # L1+0.1*Lssim
    criterion = SetCriterion(losses, weight_dict)
    model = JIIF_multiple2(16, 16, 2).cuda()
    WEIGHT_DECAY = 1e-8  # params of ADAM

    num_params = 0
    for param in JIIF_multiple2(16, 16, 2).parameters():
        num_params += param.numel()
    print('[Network %s] Total number of parameters : %.3f M' % ('JIIF_conv', num_params / 1e6))
    model.set_metrics(criterion)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    return model, criterion, optimizer, scheduler


if __name__ == '__main__':
    B, C, H, W = 1, 31, 128, 128
    scale = 4

    HR_MSI = torch.randn([B, 3, H, W]).cuda()
    LR_HSI = torch.randn([B, C, H // scale, W // scale]).cuda()
    lms = torch.randn([B, C, H, W]).cuda()

    model = JIIF_multiple2(H//scale, W//scale, 2)
    output = model(HR_MSI, lms, LR_HSI)
    print(output.shape)

    # build()


