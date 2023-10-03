import os
import scipy.io as sio
from spectral import *
dataset_path = os.path.join('C:/Users/0215/Desktop/code_NLMP/data/WV3')  # 数据集路径
data = sio.loadmat(os.path.join(dataset_path,'WV3_Rio.mat'))['gt']
spectral.settings.WX_GL_DEPTH_SIZE = 10
view_cube(data, bands=[1, 2, 3])