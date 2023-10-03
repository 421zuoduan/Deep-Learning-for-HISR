import torch
import torch.nn as nn
from torch.nn import functional as F



class DK_Conv2D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bias=True,im=True,name=1):
        super(DK_Conv2D, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = use_bias
        self.im=im
        self.name=name

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
        atw1=self.attention1(x) #1,9,256,256
        #atw2=self.attention2(x) #b,n*k*k,n_H,n_W

        if self.im==True:
            ita = torch.squeeze(atw1).permute(1, 2, 0).view(H,W,3,3).cpu().detach().numpy()#256,256,9
            iatw = torch.squeeze(atw1).permute(1, 2, 0).mean(dim=-1).cpu().detach().numpy()#256,256,1
            plt.imshow(iatw)
            plt.axis('off')
            plt.colorbar()
            fig = plt.gcf()
            fig.set_size_inches(7.0 / 3, 7.0 / 3)  # dpi = 300, output = 700*700 pixels
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.show()
            fig.savefig('T/d/avg{}.eps'.format(self.name), format='eps', transparent=True, dpi=300, pad_inches=0,bbox_inches = 'tight')
            print(ita[19,190])

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
            print(bias.squeeze(-1).squeeze(-1).squeeze(0))
            #plt.plot(bias.squeeze(-1).squeeze(-1).squeeze(0).cpu().detach().numpy())
            fig = plt.gcf()
            # fig.set_size_inches(7.0 / 3, 7.0 / 3)  # dpi = 300, output = 700*700 pixels
            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            # plt.gca().yaxis.set_major_locator(plt.NullLocator())
            # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            # plt.margins(0, 0)
            #plt.show()
            #fig.savefig('S/d8/B{}.svg'.format(self.name), format='svg', transparent=True, dpi=300, pad_inches=0,bbox_inches = 'tight')
            bias=bias.view(b,m).unsqueeze(1) #b,1,m
            bias=bias.repeat([1,n_H*n_W,1]) #b,n_H*n_W,m
            y=y+bias #b,n_H*n_W,m


        y=y.permute([0,2,1]) #b,m,n_H*n_W
        y=F.fold(y,output_size=(n_H,n_W),kernel_size=1) #b,m,n_H,n_W
        return y



class Res_Block(nn.Module):
    def __init__(self,in_planes,i,j):
        super(Res_Block, self).__init__()
        self.conv1=DK_Conv2D(in_planes,in_planes,3,1,1,im=True,name=i)
        self.relu1=nn.ReLU(inplace=True)
        self.conv2=DK_Conv2D(in_planes,in_planes,3,1,1,im=True,name=j)

    def forward(self,x):
        res=self.conv1(x)
        res=self.relu1(res)
        res=self.conv2(res)
        x=x+res
        return x


class DKNET(nn.Module):
    def __init__(self):
        super(DKNET, self).__init__()
        self.ups=nn.UpsamplingBilinear2d(scale_factor=4)

        self.head_conv=nn.Sequential(
            DK_Conv2D(34,64,3,1,1,im=True,name=1),
            nn.ReLU(inplace=True)
        )

        self.RB1 = Res_Block(64,2,3)
        self.RB2 = Res_Block(64,4,5)
        self.RB3 = Res_Block(64,6,7)

        self.tail_conv=DK_Conv2D(64,31,3,1,1,8,im=True,name=8)


    def forward(self,rgb,lrhsi):
        ushsi=self.ups(lrhsi)
        x=torch.cat([rgb,ushsi],1)
        x=self.head_conv(x)
        x=self.RB1(x)
        x=self.RB2(x)
        x=self.RB3(x)
        x=self.tail_conv(x)
        sr=ushsi+x
        return sr

if __name__ == '__main__':
    import torch
    from data import DatasetFromHdf5
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    import numpy as np

    device = torch.device('cuda:1')
    if True:
        C = 31
        # test_set = DatasetFromHdf5("G:\JF_Hu/3-multistage/test_harvard.h5")
        # num_testing = 10
        # sz = 1000

        test_set = DatasetFromHdf5("/Data/Dataset/HISR/CAVE/test_cave(with_up)x4.h5")
        num_testing = 11
        sz = 512
        testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=num_testing)
        output = np.zeros([num_testing, sz, sz, C])
        model = DKNET()
        path_checkpoint = '../01-results/DKNET_1000.pth'  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        model.load_state_dict(checkpoint)  # 加载模型可学习参数
        model = model.to(device)

        # for name, parameters in model.named_parameters():
        #     print(name, ':', parameters.size(), parameters)

        for iteration, batch in enumerate(testing_data_loader, 1):
            GT, HSI, MSI = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            for i in range(num_testing):
                HSIi = HSI[i, :, :, :]
                HSIi = HSIi[np.newaxis, :, :, :]
                MSIi = MSI[i, :, :, :]
                MSIi = MSIi[np.newaxis, :, :, :]
                print(HSIi.shape)
                with torch.no_grad():
                    outputi = model(MSIi, HSIi)
                    output[i, :, :, :] = outputi.permute([0, 2, 3, 1]).cpu().detach().numpy()



