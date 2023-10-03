"""
tensorflow/keras plot utilities for the neuron project

If you use this code, please cite 
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
CVPR 2018

Contact: adalca [at] csail [dot] mit [dot] edu
License: GPLv3
"""

# third party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from decorator import decorate
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable  # plotting
import torch
import postprocess as pp

def slices(slices_in,  # the 2D slices
           titles=None,  # list of titles
           cmaps=None,  # list of colormaps
           norms=None,  # list of normalizations
           do_colorbars=False,  # option to show colorbars on each slice
           grid=False,  # option to plot the images in a grid or a single row
           width=15,  # width in in
           show=True,  # option to actually show the plot (plt.show())
           imshow_args=None):
    '''
    plot a grid of slices (2d images)
    '''

    # input processing
    nb_plots = len(slices_in)
    for si, slice_in in enumerate(slices_in):
        assert len(slice_in.shape) == 2, 'each slice has to be 2d: 2d channels'
        slices_in[si] = slice_in.astype('float')

    def input_check(inputs, nb_plots, name):
        ''' change input from None/single-link '''
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps')
    norms = input_check(norms, nb_plots, 'norms')
    imshow_args = input_check(imshow_args, nb_plots, 'imshow_args')
    for idx, ia in enumerate(imshow_args):
        imshow_args[idx] = {} if ia is None else ia

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.title.set_text(titles[i])

        # show figure
        im_ax = ax.imshow(slices_in[i], cmap=cmaps[i], interpolation="nearest", norm=norms[i], **imshow_args[i])

        # colorbars
        # http://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        if do_colorbars and cmaps[i] is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im_ax, cax=cax)

    # clear axes that are unnecessary
    for i in range(nb_plots, col * row):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)
    plt.tight_layout()

    if show:
        plt.show()

    return (fig, axs)


def flow_legend():
    """
    show quiver plot to indicate how arrows are colored in the flow() method.
    https://stackoverflow.com/questions/40026718/different-colours-for-arrows-in-quiver-plot
    """
    ph = np.linspace(0, 2 * np.pi, 13)
    x = np.cos(ph)
    y = np.sin(ph)
    u = np.cos(ph)
    v = np.sin(ph)
    colors = np.arctan2(u, v)

    norm = Normalize()
    norm.autoscale(colors)
    # we need to normalize our colors array to match it colormap domain
    # which is [0, 1]

    colormap = cm.winter

    plt.figure(figsize=(6, 6))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.quiver(x, y, u, v, color=colormap(norm(colors)), angles='xy', scale_units='xy', scale=1)
    plt.show()


class FlowField(object):

    @classmethod
    def flow(cls, slices_in,  # the 2D slices
             flows_in,
             inpaint=True,
             mode="qv",
             titles=None,  # list of titles
             cmaps=None,  # list of colormaps
             width=15,  # width in in
             img_indexing=True,  # whether to match the image view, i.e. flip y axis
             grid=False,  # option to plot the images in a grid or a single row
             show=True,  # option to actually show the plot (plt.show())
             scale=1):  # note quiver essentially draws quiver length = 1/scale
        '''
        plot a grid of flows (2d+2 images)
        '''
        for idx, flow_in in enumerate(flows_in):
            if isinstance(flow_in, torch.Tensor):
                flows_in[idx] = torch.squeeze(flow_in.permute(1, 2, 0)).cpu().numpy()

        # input processing
        nb_plots = max(len(flows_in), len(slices_in))

        for flow_in in flows_in:
            assert len(flow_in.shape) == 3, 'each slice has to be (H, W, 2)'
            assert flow_in.shape[-1] == 2, 'each slice has x,y ndims: 2 channels'

        def input_check(inputs, nb_plots, name):
            ''' change input from None/single-link '''
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
                'number of %s is incorrect' % name
            if inputs is None:
                inputs = [None]
            if len(inputs) == 1:
                inputs = [inputs[0] for i in range(nb_plots)]
            return inputs

        if img_indexing:
            for si, slc in enumerate(flows_in):
                flows_in[si] = np.flipud(slc)  # 翻转列表、矩阵

        titles = input_check(titles, nb_plots, 'titles')
        cmaps = input_check(cmaps, nb_plots, 'cmaps')
        scale = input_check(scale, nb_plots, 'scale')

        # figure out the number of rows and columns
        if grid:
            if isinstance(grid, bool):
                rows = np.floor(np.sqrt(nb_plots)).astype(int)
                cols = np.ceil(nb_plots / rows).astype(int)
            else:
                assert isinstance(grid, (list, tuple)), \
                    "grid should either be bool or [rows,cols]"
                rows, cols = grid
        else:
            rows = 1
            cols = nb_plots

        # prepare the subplot
        if not hasattr(cls, 'fig'):
            cls.fig, cls.axs = plt.subplots(rows, cols, figsize=(10, 5))
        plt.cla()
        # cls.fig.canvas.mpl_connect('key_press_event', on_key_press)
        axs = cls.axs

        if rows == 1 and cols == 1:
            axs = [cls.axs]


        if slices_in[nb_plots-1] is not None:
            for idx, slice_in in enumerate(slices_in):
                if isinstance(slice_in, torch.Tensor):
                    slices_in[idx] = torch.squeeze(slice_in[0, ...].permute(1, 2, 0)).cpu().numpy()
            axs[0].imshow(slices_in[0], cmap='gray')
            nb_plots = nb_plots-1

        for i in range(nb_plots):
            col = np.remainder(i, cols) + 1  # 返回两数组相除后的余数
            row = np.floor(i / cols).astype(int)

            # get row and column axes
            row_axs = axs if rows == 1 else axs[row]
            ax = row_axs[col]

            # turn off axis
            ax.axis('off')

            # add titles
            if titles is not None and titles[i] is not None:
                ax.title.set_text(titles[i])


            if mode == "qv":
                u, v = flows_in[i][..., 0], flows_in[i][..., 1]
                colors = np.arctan2(u, v)
                colors[np.isnan(colors)] = 0
                norm = Normalize()
                norm.autoscale(colors)
                if cmaps[i] is None:
                    colormap = cm.winter
                else:
                    raise Exception("custom cmaps not currently implemented for plt.flow()")

                # show figure
                ax.quiver(u, v,
                          color=colormap(norm(colors).flatten()),
                          angles='xy',
                          units='xy',
                          scale=scale[i])
                ax.axis('equal')
                # ax.add_patch(q)
            elif slices_in[-1] is not None:
                ax.imshow(slices_in[nb_plots - 1], cmap='gray')
                cls.grid2contour(cls, slices_in[-1], flows_in[i], ax, axs[0],inpaint=inpaint)
            else:
                cls.grid2contour(cls, slices_in[-1], flows_in[i], ax,inpaint=inpaint)


        # clear axes that are unnecessary
        for i in range(nb_plots, col * row):
            col = np.remainder(i, cols)
            row = np.floor(i / cols).astype(int)

            # get row and column axes
            row_axs = cls.axs if rows == 1 else cls.axs[row]
            ax = row_axs[col]

            ax.axis('off')

        # show the plots
        # cls.fig.set_size_inches(width, rows / cols * width)
        plt.tight_layout()

        if show:
            plt.show()

        return (cls.fig, cls.axs)

    def grid2contour(self, I, flow_field, ax, ax_pre=None, inpaint=True):
        '''
        grid--image_grid used to show deform field
        type: numpy ndarray, shape： (h, w, 2), value range：(-1, 1)
        '''
        if not hasattr(self, 'regular_grid'):
            self.img_shape = flow_field.shape
            x = np.arange(-1, 1, 2 / self.img_shape[1])
            y = np.arange(-1, 1, 2 / self.img_shape[0])
            X, Y = np.meshgrid(x, y)
            self.regular_grid = np.stack((X, Y), axis=2)

        if inpaint:
            # if I is not None:
                # Z1 = self.regular_grid[:, :, 0] + 2  # remove the dashed line 虚线
                # Z1 = Z1[::-1]  # vertical flip
                # Z2 = self.regular_grid[:, :, 1] + 2
                # ax_pre.contour(Z1, levels=15, linewidths=1, colors='red')#ax.contour(X, Y, Z1, 15, colors='k')
                # ax_pre.contour(Z2, levels=15, linewidths=1, colors='red')#ax.contour(X, Y, Z2, 15, colors='k')
                # plt.xticks(()), plt.yticks(())  # remove x, y ticks
                # plt.title('regular field')


            # flow_field = np.random.rand(*self.img_shape)
            # flow_field = flow_field.copy()
            # flow_field[:, :, 0] = flow_field[:, :, 0] / self.img_shape[1]
            # flow_field[:, :, 1] = flow_field[:, :, 1] / self.img_shape[0]
            grid = self.regular_grid + flow_field

            assert grid.ndim == 3
            x = np.arange(-1, 1, 2 / grid.shape[1])
            y = np.arange(-1, 1, 2 / grid.shape[0])
            X, Y = np.meshgrid(x, y)
            Z1 = grid[:, :, 0] + 2  # remove the dashed line 虚线
            Z1 = Z1[::-1]  # vertical flip
            Z2 = grid[:, :, 1] + 2

            # plt.figure()
            # fig, ax = plt.subplots()
            # ax.imshow(I, cmap='gray')
            ax.contour(Z1, levels=15, linewidths=1, colors='red')
            ax.contour(Z2, levels=15, linewidths=1, colors='red')
            plt.xticks(()), plt.yticks(())  # remove x, y ticks
            plt.title('deform field')


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    X = np.arange(-10, 10, 1)
    Y = np.arange(-10, 10, 1)
    U, V = np.meshgrid(X, Y)

    fig, ax = plt.subplots()
    q = ax.quiver(X, Y, U, V,
                  angles='xy', units='xy')
    ax.quiverkey(q, X=0.3, Y=1.1, U=10,
                 label='Quiver key, length = 10', labelpos='E')

    X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
    U = np.cos(X)
    V = np.sin(Y)
    fig3, ax3 = plt.subplots()
    ax3.set_title("pivot='tip'; scales with x view")
    M = np.hypot(U, V)
    Q = ax3.quiver(X, Y, U, V, M, units='x', pivot='tip', width=0.022,
                   scale=1 / 0.15)
    qk = ax3.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                       coordinates='figure')
    ax3.scatter(X, Y, color='0.5', s=1)

    plt.show()

    plt.show()
