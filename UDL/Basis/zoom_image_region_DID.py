# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved 
#
# @Time    : 2021/10/15 17:53
# @Author  : Xiao Wu
# reference:
# https://matplotlib.org/stable/gallery/axes_grid1/inset_locator_demo.html
# https://matplotlib.org/3.2.1/api/_as_gen/mpl_toolkits.axes_grid1.inset_locator.inset_axes.html
# https://zhuanlan.zhihu.com/p/50122016

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
from matplotlib.widgets import Slider, RadioButtons
import imageio
import cv2
from typing import List
from PIL import Image
# from UDL.derain.common.derain_dataset import PSNR_ycbcr, add_mean, sub_mean
from skimage.metrics import peak_signal_noise_ratio as PSNR
from UDL.Basis.postprocess import convert_to_grayscale, norm_image

def mpl_save_fig(filename, suffix='svg'):
    plt.savefig(f"{filename}.{suffix}", format=suffix, dpi=300, pad_inches=0, bbox_inches='tight')

def plot_rectangle(ax, bbox, ls="-", line_width=2, edge_color='black', fc='None'):
    ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ls=ls, lw=line_width,
                               ec=edge_color, fc="None"))
    return ax


def show_images_Slider(dict_images, min=10):
    # H, W, C
    keys = list(dict_images.keys())
    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=(len(keys), len(keys)))
    grid = plt.GridSpec(nrows=4, ncols=4, figure=fig, height_ratios=[3] * 3 + [0.2])
    # plt.subplots_adjust(left=0.25, bottom=0.35)

    # lateral_ax = plt.axes([0.25, 0, 0.15, 0.005 * len(keys)])
    # y, x
    ax = fig.add_subplot(grid[:-1, :-1], yticks=[], xticks=[])
    lateral_ax = fig.add_subplot(grid[:, -1], yticks=[], xticks=[])
    radio = RadioButtons(lateral_ax, keys)

    def func(key):

        images = dict_images[key]
        # fig, ax = plt.subplots()
        # plt.subplots_adjust(left=0.25, bottom=0.25)
        # ax = fig.add_subplot(212)
        # axindex = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        axindex = fig.add_subplot(grid[-1, :-1], yticks=[], xticks=[])
        axindex.clear()
        sindex = Slider(axindex, 'Index', valmin=0, valmax=images.shape[-1] - 1, valinit=0, valstep=1)


        def update(val):
            index = int(sindex.val)
            print(index)
            ax.clear()
            ax.set_xlabel('index={}'.format(index))
            ax.imshow(images[..., index])
            ax.set_title(key)

            fig.canvas.draw_idle()

        sindex.on_changed(update)

        update(None)

    radio.on_clicked(func)
    plt.show()


def show_region_images(image, xywh=None, fxywh=None, dmap='rgb', edge_color='black', line_shape='-', line_width=2,
                       scale_ratio=2, sub_ax_anchor=(0, 0, 1, 1), borderpad=0):
    '''
    xywh: [x0, y0, w, h] 直接设置子图的位置和大小
    fxywh: [x0_w_ratio, y0_h_ratio, x1_w_ratio, y1_h_ratio],
           x0 = w*x0_w_ratio, y0 = h*y0_h_ratio
           x1 = w*x1_w_ratio, y1 = h*y1_h_ratio
           得到相对比全图的百分比子图

    '''
    h, w = image.shape[:2]
    sub_height = str(xywh[-1] * scale_ratio * 100 / h) + "%"
    sub_width = str(xywh[-2] * scale_ratio * 100 / w) + "%"
    # print(sub_width, sub_height)

    if all([xywh, fxywh]) is None:
        raise ValueError("given size or fxy is None")

    ################################################
    # 主图
    ################################################

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.set_xticks([])
    ax.set_yticks([])
    cmap = None
    if dmap == 'rgb':
        ax.imshow(image)  # interpolation='bilinear' , origin='lower'
    elif dmap == 'aem':
        shape = image.shape
        if len(shape) == 2:
            image = image[..., None]
            shape = image.shape
        assert len(shape) == 3 and shape[-1] in [1, 3, 4], print("image should be h, w, c")
        cmap = 'Spectral'
        ax.imshow(image, cmap=cmap)
    ################################################
    # 局部图
    ################################################
    # 嵌入坐标系

    axins = inset_axes(ax, width=sub_width, height=sub_height, loc='lower left',
                       bbox_to_anchor=sub_ax_anchor,  # [left, bottom, width, height], or [left, bottom]
                       bbox_transform=ax.transAxes,
                       borderpad=borderpad)

    # axins = ax.inset_axes(sub_ax_anchor)  # 距离左下角点的偏差
    '''
    ax：父坐标系
    width, height：百分比形式：它是子坐标系的宽度和高度相对于ax的bbox的比例；浮点数：英寸
        满足: width = bbox.width * r + a * dpi
        123.2 = 308 * 0.4 + 0 * 100
        
    loc：子坐标系的原点位置, 缺省为 upper right
        'upper right' : 1,  
        'upper left' : 2,
        'lower left' : 3,
        'lower right' : 4,
        'right' : 5,
        'center left' : 6,
        'center right' : 7,
        'lower center' : 8,
        'upper center' : 9,
        'center' : 10
    bbox_to_anchor：边界框，四元数组（x0, y0, width, height
    bbox_transform：从父坐标系到子坐标系的几何映射
    axins：子坐标系
    borderpad: 子坐标系与主坐标系之间的填充
    
    '''
    if fxywh is not None and len(fxywh) == 4:
        h, w, c = image.shape
        # [x0, y0, x1, y1]
        bbox = (
            h * fxywh[0], w * fxywh[1], w * fxywh[3], h * fxywh[2])  # (h*fxywh[0], w*fxywh[1]), w*fxywh[3], h*fxywh[2]
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[3], bbox[2], ls="-", lw=line_width,
                                   ec=edge_color, fc="None"))
        # [int(h * fxywh[0]): int(h * (fxywh[0] + fxywh[2])), int(w * fxywh[1]): int(w * (fxywh[1] + fxywh[3]))]
        axins.imshow(image[int(bbox[0]): int(bbox[0] + bbox[2]), int(bbox[1]): int(bbox[1])], cmap=cmap)
    elif xywh is not None and len(xywh) == 4:
        # (xywh[0], xywh[1]), xywh[2], xywh[3]
        bbox = xywh
        ax.add_patch(plt.Rectangle((xywh[0], xywh[1]), xywh[2], xywh[3], ls=line_shape, lw=line_width,
                                   ec=edge_color, fc="None"))
        # print(image.shape)
        # image = cv2.resize(image[xywh[1]: xywh[1] + xywh[3], xywh[0]: xywh[0] + xywh[2]], (xywh[3] * 1, xywh[2] * 1))
        # print(image.shape)
        axins.imshow(image[xywh[1]: xywh[1] + xywh[3], xywh[0]: xywh[0] + xywh[2]], cmap=cmap)  # interpolation='bilinear'
    else:
        print("bbox produce error: ", xywh, fxywh)
    # 调整子坐标系的显示范围
    axins.set_xticks([])
    axins.set_yticks([])

    # 建立子图区域与目标区域连线
    # loc1 loc2: 坐标系的四个角
    # 1 (右上) 2 (左上) 3(左下) 4(右下)
    # 法1
    # mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
    # plt.show()

    # 法2 https://zhuanlan.zhihu.com/p/147274030
    # tx0 = xlim0
    # tx1 = xlim1
    # ty0 = ylim0
    # ty1 = ylim1
    # sx = [tx0, tx1, tx1, tx0, tx0]
    # sy = [ty0, ty0, ty1, ty1, ty0]
    # ax.plot(sx, sy, "black")
    # ax.add_patch(ax.)
    # xy = (xlim0, ylim0) # xyB
    # xy2 = (xlim0, ylim1) # xyA

    # xyA是子图里面的点，xyB是主图里面的点, axesA为子图坐标，axesB为要连接的主图坐标
    # con = ConnectionPatch(xyA=(bbox[0], bbox[1]), xyB=(200, 200), coordsA="data", coordsB="data", axesA=axins, axesB=ax)
    # axins.add_artist(con)
    #
    # con = ConnectionPatch(xyA=(bbox[0]+bbox[2], bbox[1]+bbox[2]), xyB=(250, 200), coordsA="data", coordsB="data", axesA=axins, axesB=ax)
    # axins.add_artist(con)


def absoulte_error_map(abs_img: np.ndarray):
    shape = abs_img.shape

    if len(shape) == 2:
        abs_img = abs_img[..., None]
        shape = abs_img.shape

    assert len(shape) == 3 and shape[-1] in [1, 3, 4], print("image should be h, w, c")

    y = np.sum(abs_img, axis=-1).reshape((-1, 1))

    # Set up the axes with gridspec
    # constrained_layout=True,#类似于tight_layout，使得各子图之间的距离自动调整【类似excel中行宽根据内容自适应】
    fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(ncols=4, nrows=4, hspace=0.2, wspace=0)
    # 可以手动控制来合并子图, grid [y,x]
    # main_ax = fig.add_subplot(grid[:-1, 1:])
    # y_hist = fig.add_subplot(grid[:-1, 0], sharey=main_ax)
    # x_hist = None
    # x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

    main_ax = fig.add_subplot(grid[1:, :], yticklabels=[], xticklabels=[])
    x_hist = fig.add_subplot(grid[0, :])
    # x_hist = None
    y_hist = None

    # scatter points on the main axes
    # main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2)
    # main_ax.imshow(abs_img, origin='lower', cmap='gray')
    main_ax.imshow(cv2.cvtColor(abs_img, cv2.COLOR_RGB2GRAY), cmap='jet', origin='lower')
    # histogram on the attached axes
    if x_hist is not None:
        x_hist.hist(y, 40, histtype='stepfilled',
                    orientation='horizontal', color='gray')#vertical
        # x_hist.set_xlim(0, 200)
        # x_hist.set_ylim(0, 256)
        # x_hist.invert_yaxis()

    if y_hist is not None:
        y_hist.hist(y, 40, histtype='stepfilled',
                    orientation='horizontal', color='gray')
        # y_hist.set_ylim(0, 256)
        y_hist.invert_xaxis()
        # y_hist.invert_yaxis()

# 旧: 下侧放置子图
def show_down_outof_region_images(image, region_list: List[List], scale=1, dpi=72):
    dpi_default = max(72, dpi)

    fig_height = 6.
    # height = 50 + 256#length_y_axis * rows
    # width = 256#length_x_axis * columns
    # plot_aspect_ratio = float(width) / float(height)

    h, w = image.shape[:-1]
    # aspect = int(dpi / dpi_default)
    # print(aspect)
    figw = round(w * scale / dpi, 1)
    figh = round((h / dpi) * scale / 0.75, 1)

    # fig = plt.figure(tight_layout=True, figsize=(figw, figh), dpi=dpi)
    fig = plt.figure(tight_layout=True, figsize=(fig_height * 0.75, round(fig_height, 1)), dpi=dpi_default)
    print(fig.get_size_inches())  # 6*72 = 432

    # fig.set_size_inches((fig_height * plot_aspect_ratio, fig_height))
    # shape = np.shape(abs_img)
    # assert len(shape) == 3 and shape[-1] in [1, 3, 4], print("image should be h, w, c")

    # y = np.sum(abs_img, axis=-1).reshape((-1, 1))

    # Set up the axes with gridspec
    # constrained_layout=True,#类似于tight_layout，使得各子图之间的距离自动调整【类似excel中行宽根据内容自适应】
    # fig = plt.figure(figsize=(6, 6), tight_layout=True)
    # width_ratios = np.ones([4, 4])
    # width_ratios[-1, :2] = 1
    grid = plt.GridSpec(ncols=4, nrows=4, hspace=0, wspace=0, figure=fig)
    cols = grid.ncols

    # 可以手动控制来合并子图, grid [y,x]
    main_ax = fig.add_subplot(grid[:-1, :], yticks=[], xticks=[])  # (figwidth, figheight)
    # main_ax.set_aspect(0.75)

    # main_ax.set_aspect('equal')

    # bottom1 = fig.add_subplot(grid[-1, :2], yticks=[], xticks=[])
    # bottom2 = fig.add_subplot(grid[-1, 2:], yticks=[], xticks=[])
    # gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])#将以上第一个子图gs0[0]再次切片为3行3列的9个axes
    # image_t = Image.fromarray(image)
    # image_t = image_t.resize((image.shape[1], int(image.shape[0] * 0.75)), Image.ANTIALIAS)
    # print(image_t.size)
    main_ax.imshow(image)  # , aspect='auto') # extent
    # print(main_ax.get_images()[0].get_extent())
    interval = cols // len(region_list)
    idx = 0
    for bbox in region_list:
        bottom = fig.add_subplot(grid[-1, idx * interval: (idx + 1) * interval], yticks=[], xticks=[])
        # bottom.set_aspect('equal')
        # print(bottom.get_aspect())
        # bottomin = bottom.inset_axes((0, 0, 1, 1))
        # bottomin.set_xticks([])
        # bottomin.set_yticks([])
        # main_ax = plot_rectangle(main_ax, bbox)
        img = image[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]  # height, width
        # x,y : width, height
        # img.thumbnail((75, 75), Image.ANTIALIAS)  # resizes image in-place
        # bottom
        img_t = Image.fromarray(img)
        img_t = img_t.resize((img.shape[1], int(img.shape[0] * 0.66)), Image.ANTIALIAS)
        print(img_t.size)
        bottom.imshow(img_t)
        # print(bottomin.get_images()[0])

        idx += 1

# 下侧放置子图
def show_down_outof_region_images_V2(image, region_list: List[List], save_name,
                                     scale=1, dpi=72, fig_height=6, is_save=False):
    '''
    region_list : bbox [x, y, w, h] bbox[2:][::-1] == patch_size [h, w]
    所以的patch应该是一样大的 , patch_size=[50, 25]
    '''
    dpi_default = max(72, dpi)
    pixel_bsize = 1 / dpi_default
    ncols = len(region_list)
    patch_size = region_list[0][2:][::-1]
    h, w = image.shape[:2]

    figw = w
    figh = h + (patch_size[0] * w) / (ncols * patch_size[1])#w // ncols
    aspect = figw / figh

    if fig_height is not None:
        figh = fig_height
        figw = fig_height * aspect

    print(round(figw, 1) / round(figh, 1))
    fig = plt.figure(tight_layout=True, figsize=(figw * pixel_bsize, figh * pixel_bsize), dpi=dpi_default)
    print(fig.get_size_inches())  # 6*72 = 432

    # fig.set_size_inches((fig_height * plot_aspect_ratio, fig_height))
    # shape = np.shape(abs_img)
    # assert len(shape) == 3 and shape[-1] in [1, 3, 4], print("image should be h, w, c")

    # y = np.sum(abs_img, axis=-1).reshape((-1, 1))

    # Set up the axes with gridspec
    # constrained_layout=True,#类似于tight_layout，使得各子图之间的距离自动调整【类似excel中行宽根据内容自适应】
    # fig = plt.figure(figsize=(6, 6), tight_layout=True)
    # width_ratios = np.ones([4, 4])
    # width_ratios[-1, :2] = 1
    # grid = plt.GridSpec(ncols=ncols, nrows=2, hspace=0, wspace=0, figure=fig,
    #                     width_ratios=[patch_size[1] // ncols] * ncols, height_ratios=[h, w // ncols])

    grid = plt.GridSpec(ncols=ncols, nrows=2, hspace=0, wspace=0, figure=fig,
                        height_ratios=[h, (patch_size[0] * w) / (ncols * patch_size[1])])
                        # width_ratios=[patch_size[1] // ncols] * ncols, height_ratios=[h, patch_size[0]])
    # 可以手动控制来合并子图, grid [y,x]
    main_ax = fig.add_subplot(grid[:-1, :], yticks=[], xticks=[])

    # main_ax.set_aspect('equal')

    # bottom1 = fig.add_subplot(grid[-1, :2], yticks=[], xticks=[])
    # bottom2 = fig.add_subplot(grid[-1, 2:], yticks=[], xticks=[])
    # gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])#将以上第一个子图gs0[0]再次切片为3行3列的9个axes
    # image_t = Image.fromarray(image)
    # image_t = image_t.resize((image.shape[1], int(image.shape[0] * 0.75)), Image.ANTIALIAS)
    # print(image_t.size)
    main_ax.imshow(image)  # , aspect='auto') # extent
    # print(main_ax.get_images()[0].get_extent())
    interval = 1
    idx = 0
    for bbox in region_list:
        bottom = fig.add_subplot(grid[-1, idx * interval: (idx + 1) * interval], yticks=[], xticks=[])
        # bottom.set_aspect('equal')
        # print(bottom.get_aspect())
        bottomin = bottom.inset_axes((0, 0, 1, 1))
        bottomin.set_xticks([])
        bottomin.set_yticks([])
        main_ax = plot_rectangle(main_ax, bbox)
        img = image[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]  # height, width
        # x,y : width, height
        # img.thumbnail((75, 75), Image.ANTIALIAS)  # resizes image in-place
        # bottom
        img_t = Image.fromarray(img)
        img_t = img_t.resize((int(img.shape[1]), int(img.shape[0])), Image.ANTIALIAS)
        print(img_t.size)
        bottomin.imshow(img_t)
        # print(bottomin.get_images()[0])

        idx += 1
    if is_save:
        mpl_save_fig(save_name)

# 右侧放置子图，TODO: fix
def show_outof_region_images(image, region_list: List[List], save_name, patch_size=[50, 50], nrows=2, ncols=[1, 4], scale=1,
                             dpi=72, fig_height=None, is_save=False):
    dpi_default = max(72, dpi)
    pixel_bsize = 1 / dpi_default
    h, w = image.shape[:2]

    if isinstance(nrows, int) and isinstance(ncols, List):
        # h, w
        main_size = [nrows, ncols[0]]
        rest = ncols[1:] + [0]

    else:
        raise NotImplementedError
    print(main_size, rest)
    width_ratio = w / patch_size[1]
    figw = w * pixel_bsize * main_size[1] + patch_size[1] * pixel_bsize * 20
    figh = h * pixel_bsize * main_size[0]  # + patch_size[0] * pixel_bsize * rest[-1]

    if fig_height is not None:
        aspect = figw / figh
        figh = fig_height
        figw = fig_height * aspect

    print(round(figw, 1) / round(figh, 1))  # [4.29050279 6.        ]   [4.5 6. ]
    fig = plt.figure(tight_layout=True, figsize=(figw, figh), dpi=dpi_default)
    print(fig.get_size_inches())
    grid = plt.GridSpec(ncols=sum(ncols), nrows=nrows, hspace=0.1, wspace=0, figure=fig,
                        width_ratios=[3, 1, 1, 1, 1])
    # 可以手动控制来合并子图, grid [y,x]
    main_ax = fig.add_subplot(grid[:, 0], yticks=[], xticks=[])
    main_ax.imshow(image)

    idx = 1
    interval = 1
    offset = len(region_list) // 2
    for bbox in region_list[:offset]:
        top = fig.add_subplot(grid[0, idx * interval: (idx + 1) * interval], yticks=[], xticks=[])
        bottom = fig.add_subplot(grid[1, idx * interval: (idx + 1) * interval], yticks=[], xticks=[])
        top.set_title(f"{idx}_top", y=-0.1)
        bottom.set_title(f"{idx}_bottom", y=-0.1)
        main_ax = plot_rectangle(main_ax, bbox)
        top_img = image[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]  # height, width
        bbox = region_list[idx + offset - 1]
        down_img = image[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]
        main_ax = plot_rectangle(main_ax, bbox)
        bottom.imshow(down_img)
        top.imshow(top_img)
        idx += 1
    if is_save:
        mpl_save_fig(save_name)


def show_compared_image(compared_image, bbox, nrows=2, scale=1,
                             dpi=72, fig_height=None, line_width=1, is_save=False):
    dpi_default = max(72, dpi)
    hspace = 0.2
    wspace = 0
    pixel_bsize = 1 / dpi_default
    image = compared_image['input']
    h, w = image.shape[:2]
    patch_size = bbox[2:][::-1] # h, w
    offset = ncols = len(compared_image) // 2


    # width_ratio = w / patch_size[1]
    # figw = w * pixel_bsize * main_size[1] + patch_size[1] * pixel_bsize * 20
    # figh = h * pixel_bsize * main_size[0]  # + patch_size[0] * pixel_bsize * rest[-1]

    # figw = 2 * w * pixel_bsize
    # figh = h * pixel_bsize + hspace
    figw = w + ncols * ((patch_size[1] * h) / (patch_size[0] * nrows)) + wspace
    figh = h + hspace


    if fig_height is not None:
        aspect = figw / figh
        figh = fig_height
        figw = fig_height * aspect

    print(round(figw, 1) / round(figh, 1))  # [4.29050279 6.        ]   [4.5 6. ]
    fig = plt.figure(tight_layout=True, figsize=(figw * pixel_bsize * scale, figh * pixel_bsize * scale), dpi=dpi)
    print(fig.get_size_inches())
    grid = plt.GridSpec(ncols=ncols + 1, nrows=nrows, hspace=hspace, wspace=wspace, figure=fig,
                        width_ratios=[w] + [((patch_size[1] * h) / (patch_size[0] * nrows))] * ncols)# width_ratios=[w] + [w//offset] * offset
    # 可以手动控制来合并子图, grid [y,x]
    main_ax = fig.add_subplot(grid[:, 0], yticks=[], xticks=[])
    # axins = main_ax.inset_axes((0, 0, 1, 1))
    main_ax.imshow(image)

    idx = 1
    interval = 1
    assert len(compared_image) % 2 == 0

    keys = list(compared_image.keys())
    for algo in keys[:offset]:
        top = fig.add_subplot(grid[0, idx * interval: (idx + 1) * interval], yticks=[], xticks=[])
        bottom = fig.add_subplot(grid[1, idx * interval: (idx + 1) * interval], yticks=[], xticks=[])
        # bottomin = bottom.inset_axes((0, 0, 1, 1))
        # topin = top.inset_axes((0, 0, 1, 1))
        top.set_title(f"{algo}", y=-hspace) #{idx}_
        bottom.set_title(f"{keys[idx + offset - 1]}", y=-hspace)
        main_ax = plot_rectangle(main_ax, bbox, line_width=line_width)
        top_img = compared_images[algo][bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]  # height, width
        down_img = compared_images[keys[idx + offset - 1]][bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]
        main_ax = plot_rectangle(main_ax, bbox)
        bottom.imshow(down_img)
        top.imshow(top_img)
        idx += 1

    if is_save:
        mpl_save_fig("compared")

def gen_showImages(compared_images, region_list, is_save=False): #func
    # show_down_outof_region_images_V2
    for algo, img in compared_images.items():
        # func(img, algo, is_save)
        show_down_outof_region_images_V2(img, region_list, fig_height=None, save_name=algo, is_save=is_save)
        # show_down_outof_region_images(img, region_list, fig_height=None, save_name=algo, is_save=is_save)
        # show_region_images(img, region_list[0])

def gen_absImages(compared_images, region_list, is_save=False):
    assert 'gt' in compared_images.keys()
    assert len(region_list) > 0 and isinstance(region_list, list) and isinstance(region_list[0], int), print("region_list should be [x, y, w, h]")

    gt = compared_images.pop('gt')

    for idx, (algo, img) in enumerate(compared_images.items()):
        # abs = cv2.cvtColor(compared_images[algo] - compared_images['gt'], cv2.COLOR_RGB2GRAY)
        abs = np.abs(compared_images[algo] - gt)
        # abs = np.asarray(np.sqrt((img - gt) ** 2), dtype=np.uint8)
        # abs = np.where(abs < 10, abs, 0)
        # print(np.allclose(abs+compared_images[algo], compared_images['gt']))
        # abs = cv2.cvtColor(compared_images[algo], cv2.COLOR_RGB2GRAY) - cv2.cvtColor(compared_images['gt'], cv2.COLOR_RGB2GRAY)
        # abs = abs / 255.0
        print(abs.shape, np.max(abs), np.min(abs), np.mean(abs), np.std(abs))
        # plt.imshow(cv2.cvtColor(abs, cv2.COLOR_RGB2GRAY), cmap='gray')
        # plt.title("abs")
        # plt.show()
        show_region_images(compared_images[algo], region_list)
        show_region_images(abs, region_list, dmap='aem') # convert_to_grayscale(abs)


        # fig = plt.figure(figsize=(6, 6))
        # plt.hist(abs.reshape(-1), 40, histtype='stepfilled')
        ############################################################
        # plt.imshow(compared_images[algo])
        # plt.title(algo)
        # plt.show()
        # plt.imshow(compared_images['gt'])
        # plt.title("gt")
        # plt.show()
        # print(np.allclose(abs + compared_images[algo], compared_images['gt']))
        # plt.imshow(compared_images[algo] + abs)
        # plt.title("recovery")
        # plt.show()
        # absoulte_error_map(abs)
        # plt.title(algo)
        ############################################################
        if is_save:
            mpl_save_fig(algo)


def statics_PSNR(root_dir, algo_list):
    compared_files = {}
    for algo in algo_list:
        algo_dir = os.path.join(root_dir, algo, "*")
        compared_files[algo] = glob.glob(algo_dir)
        if len(compared_files[algo]) == 0:
            raise FileNotFoundError(algo)

    os.makedirs("./results_psnr_DID", exist_ok=True)

    gt_files = compared_files.pop('input')
    my_files = compared_files['DFTLW']
    psnr_dicts = {}
    print(compared_files.keys())
    for k, files in compared_files.items():
        psnr_dicts[k] = []
        print(k)
        for f, gt, my in zip(files, gt_files, my_files):
            filename = f.split('\\')[-1][:-4]
            # print(filename, psnr)
            if not os.path.isfile(f"./results_psnr_DID/{filename}_{k}.png"):
                # print(f, gt, my)
                try:
                    gt, my, img = cv2.cvtColor(cv2.imread(gt), cv2.COLOR_BGR2RGB),\
                                  cv2.cvtColor(cv2.imread(my), cv2.COLOR_BGR2RGB), \
                                  cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2RGB)
                    aem = np.abs(gt - img) #** 2
                    psnr = PSNR(gt, img)
                    # plt.imshow(aem, cmap='Spectral')
                    cv2.imwrite(f"./results_psnr_DID/{filename}_{k}_{psnr:.2f}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    # plt.savefig(f"./results_psnr_DDN3/{filename}_{k}_{psnr:.2f}.png")#, cv2.cvtColor(aem, cv2.COLOR_RGB2GRAY))
                except:
                    print(f, gt, my)
            # psnr_dicts[k].append(psnr)
        # print(k, psnr_dicts[k], np.mean(psnr_dicts[k]))

if __name__ == '__main__':
    from skimage.metrics import peak_signal_noise_ratio as PSNR
    import os
    import glob

    algo_list = ["gt", "input",  "JORDER", "Clear", "DDN", "PReNet", "RESCAN", "FBL", "FuGCN", "IPT", "DFTLW", "DFTLX"]
    # algo_list = ["gt", "input", "FuGCN", "IPT", "Restormer", "DFTL-v", "RestormerUp"]
    dataset_name = "DDN"
    root_dir = "../../../AdaConv/AdaTrans/my_model_results/DID" # 17 4 GT_p21 GT_rain_acc1 GT_rain_014 015
    for i, file in enumerate(os.listdir(root_dir+"/input")):
        print(i, file)
    image_number = [14] # 52 128

    algo_list = ["gt", "input", "DSC", "JCAS", "Clear", "DDN", "PReNet", "RESCAN", "RCDNet", "FBL", "FuGCN", "IPT", "DFTLW", "DFTLX"]#"input", "Clear", "DDN", "PReNet", "RESCAN", "RCDNet", "FuGCN", "FBL",
    # algo_list = ["gt", "input", "FuGCN", "IPT", "Restormer", "DFTL-v", "RestormerUp"]
    # dataset_name = "Rain200H"
    # root_dir = "../../../AdaConv/AdaTrans/my_model_results/Rain200H"
    image_number = [228] # 122 128 137 172 217 236 273 297 381 385
    image_number_copy = image_number
    # algo_list = ["input", "DFTLW", "DFTLX", "FBL", "IPT", "Restormer"]
    # statics_PSNR(root_dir, algo_list)

    compared_images = {}
    for algo in algo_list:
        algo_dir = os.path.join(root_dir, algo, "*")
        algo_image_files = glob.glob(algo_dir)
        if len(algo_image_files) == 0:
            raise FileNotFoundError(algo)
        if len(algo_image_files) == 1:
            image_number = [1]
        else:
            image_number = image_number_copy
        for num in image_number:
            print(algo_image_files[num - 1])
            compared_images[algo] = cv2.cvtColor(cv2.imread(algo_image_files[num - 1]), cv2.COLOR_BGR2RGB)
            # compared_images[algo] = cv2.cvtColor(cv2.imread("../../../new_data6.png"), cv2.COLOR_BGR2RGB)
            # show_down_outof_region_images_V2(compared_images[algo],
            #                                  [[0, 20, 50, 50], [120, 60, 50, 50],
            #                                   [120, 120, 50, 50], [120, 120, 50, 50]], fig_height=None, save_name=algo)

            # plt.imshow(compared_images[algo])
            # plt.title(algo)
            # plt.show()
            # show_region_images(compared_images[algo], xywh=[0, 200, 50, 50])

            # plt.show()
    print(compared_images.keys())
    ##########################################################################################
    # image = cv2.imread("../../../new_data6.png")
    # output = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=1.0)
    # abs = output - image
    # print(image.shape)
    # show_region_images(image, xywh=[0, 200, 50, 50])
    # absoulte_error_map(cv2.cvtColor(abs, cv2.COLOR_RGB2GRAY)[..., None])
    # show_down_outof_region_images_V2(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), [[0, 20, 50, 50], [120, 60, 50, 50]])
    # show_images_Slider([image, output, abs])

    # show_down_outof_region_images_V2(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), [[0, 20, 50, 50], [120, 60, 50, 50]])

    ############################################################


    # gen_absImages(compared_images, [0, 20, 50, 50], is_save=True)
    # X, Y, W, H
    # gen_showImages(compared_images, [
    #                                   [120, 120, 80, 30], [240, 60, 80, 30]], is_save=False) #[0, 20, 50, 50], [120, 60, 50, 50],
    show_compared_image(compared_images,
                        [100, 150, 200, 200], # x, y, w, h
                        dpi=300, scale=1, is_save=True)
    # gen_showImages(compared_images, [
    #     [120, 120, 80, 30], [30, 50, 80, 30]], is_save=False)
    plt.show()
