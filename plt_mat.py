import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# 读取MAT文件
mat_data = scipy.io.loadmat('my_model_results/Bidi_kernelattentionv2/AdaTrans/cave_x4/Bidi_kernelattentionv2_cave2000.mat')

# 假设MAT文件中的图像存储在变量 'images' 中
images = mat_data['output']

# 取出要作为RGB通道的通道索引（第31/20/10个通道）
channel_indices = [30, 19, 9]
# channel_indices = [1, 12, 22]

# 创建一个2x6的子图布局，以容纳所有图片
num_images = images.shape[0]
rows, cols = 2, 6
fig, axs = plt.subplots(rows, cols, figsize=(12, 6))

# 遍历每张图片并将其显示在子图中
for i in range(num_images):
    row, col = divmod(i, cols)  # 计算子图的行和列
    rgb_channels = [images[i, :, :, index] for index in channel_indices]
    rgb_image = np.stack(rgb_channels, axis=-1)
    
    axs[row, col].imshow(rgb_image)
    axs[row, col].set_title(f'Image {i + 1}')
    axs[row, col].axis('off')

plt.tight_layout()
plt.show()
