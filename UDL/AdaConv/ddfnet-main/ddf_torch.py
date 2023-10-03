from torch import nn

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