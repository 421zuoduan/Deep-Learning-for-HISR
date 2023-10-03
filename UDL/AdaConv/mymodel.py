import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter




class DK_Conv2D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bias=True):
        super(DK_Conv2D, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = use_bias

        self.attention1=nn.Sequential(
            nn.Conv2d(in_planes, kernel_size**2, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size**2,kernel_size**2,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size ** 2, kernel_size ** 2, 1),
            nn.Sigmoid()
        ) #b,9,H,W È«Í¨µÀÏñËØ¼¶ºË×¢ÒâÁ¦
        # self.attention2=nn.Sequential(
        #     nn.Conv2d(in_planes,(kernel_size**2)*in_planes,kernel_size, stride, padding,groups=in_planes),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d((kernel_size**2)*in_planes,(kernel_size**2)*in_planes,1,groups=in_planes),
        #     nn.Sigmoid()
        # ) #b,9n,H,W µ¥Í¨µÀÏñËØ¼¶ºË×¢ÒâÁ¦
        if use_bias==True:
            self.attention3=nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_planes,out_planes,1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_planes,out_planes,1)
            ) #b,m,1,1 Í¨µÀÆ«ÖÃ×¢ÒâÁ¦

        conv1=nn.Conv2d(in_planes,out_planes,kernel_size,stride,padding,dilation,groups)
        self.weight=conv1.weight # m, n, k, k


    def forward(self,x):
        (b, n, H, W) = x.shape
        m=self.out_planes
        k=self.kernel_size
        n_H = 1 + int((H + 2 * self.padding - k) / self.stride)
        n_W = 1 + int((W + 2 * self.padding - k) / self.stride)
        atw1=self.attention1(x) #b,k*k,n_H,n_W
        #atw2=self.attention2(x) #b,n*k*k,n_H,n_W

        atw1=atw1.permute([0,2,3,1]) #b,n_H,n_W,k*k
        atw1=atw1.unsqueeze(3).repeat([1,1,1,n,1]) #b,n_H,n_W,n,k*k
        atw1=atw1.view(b,n_H,n_W,n*k*k) #b,n_H,n_W,n*k*k

        #atw2=atw2.permute([0,2,3,1]) #b,n_H,n_W,n*k*k

        atw=atw1#*atw2 #b,n_H,n_W,n*k*k
        atw=atw.view(b,n_H*n_W,n*k*k) #b,n_H*n_W,n*k*k
        atw=atw.permute([0,2,1]) #b,n*k*k,n_H*n_W

        kx=F.unfold(x,kernel_size=k,stride=self.stride,padding=self.padding) #b,n*k*k,n_H*n_W
        atx=atw*kx #b,n*k*k,n_H*n_W

        atx=atx.permute([0,2,1]) #b,n_H*n_W,n*k*k
        atx=atx.view(1,b*n_H*n_W,n*k*k) #1,b*n_H*n_W,n*k*k

        w=self.weight.view(m,n*k*k) #m,n*k*k
        w=w.permute([1,0]) #n*k*k,m
        y=torch.matmul(atx,w) #1,b*n_H*n_W,m
        y=y.view(b,n_H*n_W,m) #b,n_H*n_W,m
        if self.bias==True:
            bias=self.attention3(x) #b,m,1,1
            bias=bias.view(b,m).unsqueeze(1) #b,1,m
            bias=bias.repeat([1,n_H*n_W,1]) #b,n_H*n_W,m
            y=y+bias #b,n_H*n_W,m

        y=y.permute([0,2,1]) #b,m,n_H*n_W
        y=F.fold(y,output_size=(n_H,n_W),kernel_size=1) #b,m,n_H,n_W
        return y

class DDF_Conv2D(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, dilation=1, head=1, se_ratio=0.2,
                 nonlinearity='relu', gen_kernel_size=1):
        super(DDF_Conv2D, self).__init__()
        assert kernel_size > 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.head = head
        self.mid_channels = int(in_channels * se_ratio)
        # self.kernel_combine = kernel_combine

        self.spatial_branch = nn.Sequential(
            nn.Conv2d(in_channels, kernel_size ** 2, 1),
            nn.Sigmoid()
        )
        self.channel_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, self.mid_channels, 1),
            nn.ReLU(True),
            nn.Conv2d(self.mid_channels, in_channels * kernel_size ** 2, 1),
            nn.Sigmoid()
        )

        self.unfold1 = nn.Unfold(kernel_size=self.kernel_size, dilation=1, padding=1, stride=1)

    def forward(self, x):
        b, c, h, w = x.shape
        g = self.head
        k = self.kernel_size
        s = self.stride
        channel_filter = self.channel_branch(x).reshape(b, c, k * k, 1, 1)
        spatial_filter = self.spatial_branch(x).reshape(b, 1, -1, h, w)
        # x = x.reshape(b, c, h, w)
        filter = spatial_filter * channel_filter
        # print(filter.shape)
        unfold_feature = self.unfold1(x).reshape([b, c, -1, h, w])
        out = (unfold_feature * filter).sum(2)

        return out

# --------------------------------Bias BlueConv Block -----------------------------------#
class BSConv_U_bias(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(BSConv_U_bias, self).__init__()
        self.in_planes = in_channels
        self.out_planes = out_channels
        self.kernel_size = kernel_size

        self.point_wise = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=True)

        self.point_wise_bias = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=True)

        self.depth_wise = nn.Conv2d(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=(kernel_size - 1) // 2,
                                    groups=out_channels,
                                    bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  #
        out_tmp = self.point_wise(x)  #
        out_tmp = self.depth_wise(out_tmp)  #
        bias = self.point_wise_bias(x)
        out = out_tmp + bias

        return out

# --------------------------------Res Block -----------------------------------#
class Res_Block(nn.Module):
    def __init__(self,in_planes):
        super(Res_Block, self).__init__()
        self.conv1=DDF_Conv2D(in_planes,3,1,1)
        self.relu1=nn.ReLU(inplace=True)
        self.conv2=DDF_Conv2D(in_planes,3,1,1)

    def forward(self,x):
        res=self.conv1(x)
        res=self.relu1(res)
        res=self.conv2(res)
        x=x+res
        return x


class DKNET(nn.Module):
    def __init__(self):
        super(DKNET, self).__init__()
        self.head_conv=nn.Sequential(
            DDF_Conv2D(9,3,1,1),
            # nn.Conv2d(9,32,3,1,1),
            BSConv_U_bias(9,32,3),
            nn.ReLU(inplace=True)
        )

        self.CB=nn.Sequential(
            DDF_Conv2D(32, 3, 1, 1),
            nn.ReLU(inplace=True),
            DDF_Conv2D(32, 3, 1, 1),
            nn.ReLU(inplace=True),
            DDF_Conv2D(32, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.res_block = nn.Sequential(
            Res_Block(32),
            Res_Block(32)

        )

        self.tail_conv=nn.Sequential(
            # nn.Conv2d(32,8,3,1,1),
            BSConv_U_bias(32, 8, 3),
            DDF_Conv2D(8,3,1,1)
        )

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self,pan,lms):
        x=torch.cat([pan,lms],1)
        x=self.head_conv(x)
        # x=self.CB(x)
        x = self.res_block(x)
        x=self.tail_conv(x)
        sr=lms+x
        return sr


if __name__ == '__main__':
    from torchsummary import summary
    N=DKNET()
    summary(N,[(1,64,64),(8,64,64)],device='cpu')


