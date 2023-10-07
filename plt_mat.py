import scipy.io
import matplotlib.pyplot as plt

# 读取MAT文件
mat_data = scipy.io.loadmat('my_model_results\Bidi_merge1_xca_group1_light_48\AdaTrans\cave_x4\Bidi_merge1_xca_group1_light_48_cave2000.mat')
# mat_data = scipy.io.loadmat('my_model_results\Bidi_kernelattentionv2\AdaTrans\cave_x4\Bidi_kernelattentionv2_cave2000.mat')

# 假设MAT文件中的图像存储在变量 'images' 中
images = mat_data['output']

# 显示图像
num_images = images.shape[0]

# 创建一个2x6的子图布局，可以根据需要调整行数和列数
rows, cols = 2, 6
plt.figure(figsize=(12, 6))

for i in range(num_images):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(images[i, :, :, 0], cmap='R-1')  # 显示第一个通道的图像
    plt.title(f'Image {i + 1}')
    plt.axis('off')

plt.tight_layout()
plt.show()
