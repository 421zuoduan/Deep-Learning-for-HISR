import numpy as np
import torch.nn as nn
import torch
from edsr import make_edsr_baseline, make_coord
import torch.nn.functional as F
from UDL.pansharpening.common.evaluate import analysis_accu
from UDL.Basis.criterion_metrics import *

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

def patch_mean(x):
    x0 = x[:, :, 0::2, 0::2]  # B H/2 W/2 C
    x1 = x[:, :, 1::2, 0::2]  # B H/2 W/2 C
    x2 = x[:, :, 0::2, 1::2]  # B H/2 W/2 C
    x3 = x[:, :, 1::2, 1::2]  # B H/2 W/2 C
    x = (x0 + x1 + x2 + x3)/4
    return x

def Bilateral_Filters(coord, q_coord, feat, q_feat, sigmaSpace=1, sigmaColor=10):
    ### coord && q_coord ###
    ### hr_guide && q_feat ###
    delt = 0.01
    # space_coeff = -0.5 / (sigmaSpace * sigmaSpace)
    # color_coeff = 0.5 / (sigmaColor * sigmaColor)
    #
    # G_space = space_coeff*torch.sum((coord - q_coord)*(coord - q_coord), dim=-1)
    G_color = torch.sum(torch.mul(feat[..., -4:-1], q_feat[..., -4:-1]), dim=-1)

    # weight = G_space*G_color
    # weight_ = G_space + G_color

    return G_color

class BF_NIR_conv(nn.Module):
    def __init__(self, feat_dim=128, guide_dim=128, spa_edsr_num=4, spe_edsr_num=4, mlp_dim=[256, 128], NIR_dim=32):
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.mlp_dim = mlp_dim
        self.NIR_dim = NIR_dim

        self.spatial_encoder = make_edsr_baseline(n_resblocks=spa_edsr_num, n_feats=self.guide_dim, n_colors=34)
        self.spectral_encoder = make_edsr_baseline(n_resblocks=spe_edsr_num, n_feats=self.feat_dim, n_colors=31)
        self.spatial_encoder_lr = nn.AdaptiveMaxPool2d((128, 128))

        imnet_in_dim = self.feat_dim + self.guide_dim*2 + 2

        self.imnet = MLP(imnet_in_dim, out_dim=NIR_dim, hidden_list=self.mlp_dim)
        self.decoder = nn.Sequential(nn.Conv2d(NIR_dim, NIR_dim, kernel_size=3, padding=1, bias=False),
                                    nn.Conv2d(NIR_dim, 31, kernel_size=5, padding=2, bias=False))
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

        q_guide_hr_ = hr_guide.view(b, c, -1).permute(0, 2, 1) 

        rx = 1 / h
        ry = 1 / w

        preds = []
        G_space_ = []
        G_color_ = []
        r_coord = torch.zeros_like(coord)
        r_q_coord = torch.zeros_like(coord)

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                feat_ = F.grid_sample(feat, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]

                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                r_coord[..., 0] = coord[..., 0] * h
                r_coord[..., 1] = coord[..., 1] * w

                r_q_coord[..., 0] = q_coord[..., 0] * h
                r_q_coord[..., 1] = q_coord[..., 1] * w

                inp = torch.cat([q_feat, q_guide_hr_, rel_coord], dim=-1)

                pred = self.imnet(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred)

                ### calculate the Bilateral_Filters ###
                ### coord && q_coord ###
                ### hr_guide && q_feat ###

                G_color = Bilateral_Filters(r_coord, r_q_coord, feat_, q_feat)
                G_color_.append(G_color)

        preds = torch.stack(preds, dim=-1)  # [B, N, 2, kk] # [B, N, kk]
        weight = F.softmax(torch.stack(G_color_, dim=-1), dim=-1)
        ret = (preds * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1)

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
        lr_spe = self.spectral_encoder(LR_HSI)  # Bx128xhxw The feature map of LR-HSI

        NIR_feature = self.query(lr_spe, coord, hr_spa, lr_spa)  # BxCxHxW

        # next, we can use the Transformer as the decoder
        output = self.decoder(NIR_feature)

        output = lms + output

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
    weight_dict = {'Loss': 1}
    losses = {'Loss': loss1}
    criterion = SetCriterion(losses, weight_dict)
    model = BF_NIR_conv(128, 128).cuda()
    WEIGHT_DECAY = 1e-8  # params of ADAM

    num_params = 0
    for param in BF_NIR_conv(128, 128).parameters():
        num_params += param.numel()
    print('[Network %s] Total number of parameters : %.3f M' % ('JIIF_conv', num_params / 1e6))
    model.set_metrics(criterion)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    return model, criterion, optimizer, scheduler

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    model = BF_NIR_conv(64, 64).cuda()

    B, C, H, W = 1, 31, 64, 64
    scale = 4

    HR_MSI = torch.randn([B, 3, H, W]).cuda()
    lms = torch.randn([B, C, H, W]).cuda()
    LR_HSI = torch.randn([B, C, H // scale, W // scale]).cuda()

    output = model(HR_MSI, lms, LR_HSI)
    print(output.shape)

    print(flop_count_table(FlopCountAnalysis(model, (HR_MSI, lms, LR_HSI))))
    ### 0.823M                 | 3.1G  ###
    # a = 3.85*1.18+1.53*3.58+1.6*0.97+1.37*1.15+2.79*2.76+2.52*2.72+1.53*1.57+1.8*0.95+4.16*4.11+3.02*2.46+1.65*1.87+0.4*1.16+1.06*0.44+1.77*2.38
    # print(a)