import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from torchsummary import summary
from collections import OrderedDict

def init_weights(modules):
    pass

def conv_bn_block(input, output, kernel_size):
    '''
    标准卷积块（conv + bn + relu）
    :param input: 输入通道数
    :param output: 输出通道数
    :param kernel_size: 卷积核大小
    :return:
    '''
    return nn.Sequential(
        nn.Conv1d(input, output, kernel_size),
        nn.BatchNorm1d(output),
        nn.ReLU(inplace=True)
    )


def fc_bn_block(input, output):
    '''
    标准全连接块（fc + bn + relu）
    :param input:  输入通道数
    :param output:  输出通道数
    :return:  卷积核大小
    '''
    return nn.Sequential(
        nn.Linear(input, output),
        nn.BatchNorm1d(output),
        nn.ReLU(inplace=True)
    )

class EdgeConv(nn.Module):
    '''
    EdgeConv模块
    1. 输入为：n * f
    2. 创建KNN graph，变为： n * k * f
    3. 接上若干个mlp层：a1, a2, ..., an
    4. 最终输出为：n * k * an
    5. 全局池化，变为： n * an
    '''
    def __init__(self, layers, K=20):
        '''
        构造函数
        :param layers: e.p. [3, 64, 64, 64]
        :param K:
        '''
        super(EdgeConv, self).__init__()

        self.K = K
        self.layers = layers
        # self.KNN_Graph = torch.zeros(Args.batch_size, 2048, self.K, self.layers[0]).to(Args.device)

        if layers is None:
            self.mlp = None
        else:
            mlp_layers = OrderedDict()
            for i in range(len(self.layers) - 1):
                if i == 0:
                    mlp_layers['conv_bn_block_{}'.format(i + 1)] = conv_bn_block(2*self.layers[i], self.layers[i + 1], 1)
                else:
                    mlp_layers['conv_bn_block_{}'.format(i+1)] = conv_bn_block(self.layers[i], self.layers[i+1], 1)
            self.mlp = nn.Sequential(mlp_layers)

    def createSingleKNNGraph(self, X):
        '''
        generate a KNN graph for a single point cloud
        :param X:  X is a Tensor, shape: [N, F]
        :return: KNN graph, shape: [N, K, F]
        '''
        N, F = X.shape
        assert F == self.layers[0]

        # self.KNN_Graph = np.zeros(N, self.K)

        # 计算距离矩阵
        dist_mat = torch.pow(X, 2).sum(dim=1, keepdim=True).expand(N, N) + \
                   torch.pow(X, 2).sum(dim=1, keepdim=True).expand(N, N).t()
        dist_mat.addmm_(1, -2, X, X.t())

        # 对距离矩阵排序
        dist_mat_sorted, sorted_indices = torch.sort(dist_mat, dim=1)
        # print(dist_mat_sorted)

        # 取出前K个（除去本身）
        knn_indexes = sorted_indices[:, 1:self.K+1]
        # print(sorted_indices)

        # 创建KNN图
        knn_graph = X[knn_indexes]

        return knn_graph

    def forward(self, X):
        '''
        前向传播函数
        :param X:  shape: [B, N, F]
        :return:  shape: [B, N, an]
        '''
        # print(X.shape)
        B, N, F = X.shape
        assert F == self.layers[0]

        KNN_Graph = torch.zeros(B, N, self.K, self.layers[0]).cuda()

        # creating knn graph
        # X: [B, N, F]
        for idx, x in enumerate(X):
            # x: [N, F]
            # knn_graph: [N, K, F]
            # self.KNN_Graph[idx] = self.createSingleKNNGraph(x)
            KNN_Graph[idx] = self.createSingleKNNGraph(x)
        # print(self.KNN_Graph.shape)
        # print('KNN_Graph: {}'.format(KNN_Graph[0][0]))

        # X: [B, N, F]
        x1 = X.reshape([B, N, 1, F])
        x1 = x1.expand(B, N, self.K, F)
        # x1: [B, N, K, F]

        x2 = KNN_Graph - x1
        # x2: [B, N, K, F]

        x_in = torch.cat([x1, x2], dim=3)
        # x_in: [B, N, K, 2*F]
        x_in = x_in.permute(0, 3, 1, 2)
        # x_in: [B, 2*F, N, K]

        # reshape, x_in: [B, 2*F, N*K]
        x_in = x_in.reshape([B, 2 * F, N * self.K])

        # out: [B, an, N*K]
        out = self.mlp(x_in)
        _, an, _ = out.shape
        # print(out.shape)

        out = out.reshape([B, an, N, self.K])
        # print(out.shape)
        # reshape, out: [B, an, N, K]
        out = out.reshape([B, an*N, self.K])
        # print(out.shape)
        # reshape, out: [B, an*N, K]
        out = nn.MaxPool1d(self.K)(out)
        # print(out.shape)
        out = out.reshape([B, an, N])
        # print(out.shape)
        out = out.permute(0, 2, 1)
        # print(out.shape)

        return out

class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


class Net2(nn.Module):
    def __init__(self, upscale_factor):
        super(Net2, self).__init__()

        self.conv1 = BasicBlock(1,32)
        self.conv2 = BasicBlock(32,64)
        self.conv3 = BasicBlock(64,32)
        self.conv4 = BasicBlock(32,1)
        self.gcn1 = EdgeConv(layers=[1,16,16,1], K=20)
        self.gcn2 = EdgeConv(layers=[1,16,16,1], K=20)
        self.gcn3 = EdgeConv(layers=[1,16,16,1], K=20)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        F1= torch.reshape(x, shape=[-1,x.shape[1],x.shape[2]*x.shape[3]])
        F1 = F1.permute(0,2,1)
        x2=  self.gcn1(F1)
        x2 = x2.permute(0,2,1)
        x2 = torch.reshape(x2, shape=[-1,x.shape[1],x.shape[2],x.shape[3]])

        x3=  self.gcn2(F1)
        x3 = x3.permute(0,2,1)
        x3 =  torch.reshape(x3, shape=[-1,x.shape[1],x.shape[2],x.shape[3]])

        x4=  self.gcn3(F1)
        x4 = x4.permute(0,2,1)
        x4 =  torch.reshape(x4, shape=[-1,x.shape[1],x.shape[2],x.shape[3]])

        p_x= torch.cat([x,x2,x3,x4],dim=1)
        x = self.pixel_shuffle(p_x)
        return x




class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 16 , (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(16, 1, (3, 3), (1, 1), (1, 1))
        self.gcn1 = EdgeConv(layers=[1,16,1], K=20)
        self.gcn2 = EdgeConv(layers=[1,16,1], K=20)
        self.gcn3 = EdgeConv(layers=[1,16,1], K=20)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = self.conv4(x)
        F1= torch.reshape(x, shape=[-1,x.shape[1],x.shape[2]*x.shape[3]])
        F1 = F1.permute(0,2,1)
        x2=  self.gcn1(F1)
        x2 = x2.permute(0,2,1)
        x2 = torch.reshape(x2, shape=[-1,x.shape[1],x.shape[2],x.shape[3]])

        x3=  self.gcn2(F1)
        x3 = x3.permute(0,2,1)
        x3 =  torch.reshape(x3, shape=[-1,x.shape[1],x.shape[2],x.shape[3]])

        x4=  self.gcn3(F1)
        x4 = x4.permute(0,2,1)
        x4 =  torch.reshape(x4, shape=[-1,x.shape[1],x.shape[2],x.shape[3]])

        p_x= torch.cat([x,x2,x3,x4],dim=1)
        x = self.pixel_shuffle(p_x)
        x = F.sigmoid(x)
        return x


if __name__ == "__main__":
    model = Net(upscale_factor=3)
    print(model)
