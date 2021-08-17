"""
Copyright (c) 2017, Gavin Weiguang Ding
All rights reserved.

Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
"""

import matplotlib.patches as mpatches
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle

NumDots = 4
NumConvMax = 8
NumFcMax = 20
White = 1.
Light = 0.7
Medium = 0.5
Dark = 0.4
Darker = 0.15
Black = 0.


def add_layer(patches, colors, size=(24, 24), num=5,
              top_left=[0, 0],
              loc_diff=[3, -3],
              ):
    # add a rectangle
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    for ind in range(num):
        patches.append(Rectangle(loc_start + ind * loc_diff, size[1], size[0]))
        if ind % 2:
            colors.append(Medium)
        else:
            colors.append(Light)



def add_layer_with_omission(patches, colors, size=(24, 24),
                            num=5, num_max=8,
                            num_dots=4,
                            top_left=[0, 0],
                            loc_diff=[3, -3],
                            ):
    # add a rectangle
    top_left = np.array(top_left)

    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    this_num = min(num, num_max)
    start_omit = (this_num - num_dots) // 2
    end_omit = this_num - start_omit
    start_omit -= 1
    for ind in range(this_num):
        if (num > num_max) and (start_omit < ind < end_omit):
            omit = True
        else:
            omit = False

        if omit:
            patches.append(
                Circle(loc_start + ind * loc_diff + np.array(size) / 2, 0.5))
        else:
            patches.append(Rectangle(loc_start + ind * loc_diff,
                                     size[1], size[0]))

        if omit:
            colors.append(Black)
        elif ind % 2:
            colors.append(Medium)
        else:
            colors.append(Light)




def add_mapping(patches, colors, start_ratio, end_ratio, patch_size, ind_bgn,
                top_left_list, loc_diff_list, num_show_list, size_list):

    start_loc = top_left_list[ind_bgn] \
        + (num_show_list[ind_bgn] - 1) * np.array(loc_diff_list[ind_bgn]) \
        + np.array([start_ratio[0] * (size_list[ind_bgn][1] - patch_size[1]),
                    - start_ratio[1] * (size_list[ind_bgn][0] - patch_size[0])]
                   )




    end_loc = top_left_list[ind_bgn + 1] \
        + (num_show_list[ind_bgn + 1] - 1) * np.array(
            loc_diff_list[ind_bgn + 1]) \
        + np.array([end_ratio[0] * size_list[ind_bgn + 1][1],
                    - end_ratio[1] * size_list[ind_bgn + 1][0]])


    patches.append(Rectangle(start_loc, patch_size[1], -patch_size[0],lw=0.5))
    colors.append(Dark)
    # patches.append(Line2D([start_loc[0], end_loc[0]],
    #                       [start_loc[1], end_loc[1]],lw=0.6,ls='--'))
    colors.append(Darker)
    patches.append(Line2D([start_loc[0] + patch_size[1], end_loc[0]],
                          [start_loc[1], end_loc[1]],lw=0.6,ls='--'))
    # colors.append(Darker)
    # patches.append(Line2D([start_loc[0], end_loc[0]],
    #                       [start_loc[1] - patch_size[0], end_loc[1]],lw=0.6,ls='--'))
    colors.append(Darker)
    patches.append(Line2D([start_loc[0] + patch_size[1], end_loc[0]],
                          [start_loc[1] - patch_size[0], end_loc[1]],lw=0.6,ls='--'))

    colors.append(Dark)
plt.rc('font',family='Times New Roman')
import matplotlib
# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager._rebuild()

def label(xy, text, xy_off=[0, 4]):
    plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
             family='sans-serif', size=6)


if __name__ == '__main__':

    fc_unit_size = 2
    layer_width = 40
    flag_omit = True

    patches = []
    colors = []

    fig, ax = plt.subplots()


    ############################
    # conv layers
    boxsize = [(100,100),(100,100),(50,50),(50,50),(25,25)]
    boxnumber = [1,16,16,32,32]
    size_list = [(20,20), (20, 20), (16, 16), (16, 16), (8, 8)]
    num_list = [1, 32, 32, 48, 48]

    x_diff_list = [0, layer_width, layer_width, layer_width, layer_width]
    text_list = [''] + [''] * (len(size_list) - 1)
    loc_diff_list = [[0.5, -3]] * len(size_list)

    num_show_list = list(map(min, num_list, [NumConvMax] * len(num_list)))
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]
    top_left_list[0][1] = -10

    titlelist=['     input','   layer 1','   layer 2','   layer 3','   layer 4']
    for ind in range(len(size_list)-1,-1,-1):

        top_left_list[4][1] = top_left_list[4][1] - 5


        if flag_omit:
            add_layer_with_omission(patches, colors, size=size_list[ind],
                                    num=num_list[ind],
                                    num_max=NumConvMax,
                                    num_dots=NumDots,
                                    top_left=top_left_list[ind],
                                    loc_diff=loc_diff_list[ind])
        else:
            add_layer(patches, colors, size=size_list[ind],
                      num=num_show_list[ind],
                      top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        if ind==0:
            top_left_list[0][1] = 0
        top_left_list[4][1] = top_left_list[4][1] + 5

        top_left_list[0][0] = top_left_list[0][0]-1
        top_left_list[1][0] = top_left_list[1][0] - 2
        top_left_list[2][0] = top_left_list[2][0] -2
        top_left_list[3][0] = top_left_list[3][0] - 2
        top_left_list[4][0] = top_left_list[4][0] - 5
        label(top_left_list[ind], titlelist[ind] + '\n{}x({}x{})\n'.format(
            boxnumber[ind], boxsize[ind][0], boxsize[ind][1]))
        top_left_list[0][0] = top_left_list[0][0]+1
        top_left_list[1][0] = top_left_list[1][0] + 2
        top_left_list[2][0] = top_left_list[2][0] +2
        top_left_list[3][0] = top_left_list[3][0] + 2
        top_left_list[4][0] = top_left_list[4][0] + 5
    ############################
    # in between layers
    top_left_list[0][1] = -10
    start_ratio_list = [[0.4, 0.5], [0.4, 0.8], [0.4, 0.5], [0.4, 0.8]]
    end_ratio_list = [[0.4, 0.5], [0.4, 0.8], [0.4, 0.5], [0.4, 0.8]]
    patch_size_list = [(5, 5), (2, 2), (5, 5), (2, 2)]
    ind_bgn_list = range(len(patch_size_list))
    text_list = ['Conv.', 'MaxP.', 'Conv.', 'MaxP.']

    for ind in range(len(patch_size_list)):
        add_mapping(
            patches, colors, start_ratio_list[ind], end_ratio_list[ind],
            patch_size_list[ind], ind,
            top_left_list, loc_diff_list, num_show_list, size_list)

        if ind == 0:
            top_left_list[0][1] = 0

        if ind ==0 or ind ==2 :
            label(top_left_list[ind], text_list[ind] + '\n {}x{}\n'.format(
                patch_size_list[ind][0], patch_size_list[ind][1]), xy_off=[23, -20]
                  )
        else:
            label(top_left_list[ind], text_list[ind] + '\n {}x{}\n'.format(
            patch_size_list[ind][0], patch_size_list[ind][1]), xy_off=[23, -20]
        )


    ############################
    # fully connected layers
    size_list = [(fc_unit_size, fc_unit_size)] * 3
    num_list = [20000, 800, 2]
    num_show_list = list(map(min, num_list, [NumFcMax] * len(num_list)))
    x_diff_list = [sum(x_diff_list) + layer_width, layer_width, layer_width]
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]
    loc_diff_list = [[0.2*fc_unit_size, -fc_unit_size]] * len(top_left_list)
    text_list = [''] * (len(size_list) - 1) + ['']
    # top_left_list[ind]
    unittitlelist=['layer 5','layer 6','output']
    for ind in range(len(size_list)):

        if ind==1:
            add_layer_with_omission(patches, colors, size=size_list[ind],
                                    num=num_list[ind],
                                    num_max=10,
                                    num_dots=NumDots,
                                    top_left=[240,-10],
                                    loc_diff=loc_diff_list[ind])

        else:
            if flag_omit:
                if ind == 2:
                    top_left_list[ind][1] = -18
                add_layer_with_omission(patches, colors, size=size_list[ind],
                                        num=num_list[ind],
                                        num_max=NumFcMax,
                                        num_dots=NumDots,
                                        top_left=top_left_list[ind],
                                        loc_diff=loc_diff_list[ind])


            else:
                add_layer(patches, colors, size=size_list[ind],
                          num=num_show_list[ind],
                          top_left=top_left_list[ind],
                          loc_diff=loc_diff_list[ind])

        top_left_list[ind][1] = 0
        top_left_list[2][0] = top_left_list[2][0] - 8
        top_left_list[1][0] = top_left_list[1][0] - 6
        if ind==2:
            label(top_left_list[ind], unittitlelist[ind] + '\n    {}\n'.format(
                num_list[ind]))
        elif ind==1:
            label(top_left_list[ind], unittitlelist[ind] + '\n  {}\n'.format(
                num_list[ind]))
        else:
            label(top_left_list[ind], unittitlelist[ind] + '\n{}\n'.format(
                num_list[ind]))

        top_left_list[1][0] = top_left_list[1][0] +6
        top_left_list[2][0] = top_left_list[2][0] +8


    text_list = ['Drop.&\nFlatten', 'Fully\nconn.', 'Fully\nconn.']

    for ind in range(len(size_list)):
        label(top_left_list[ind], text_list[ind], xy_off=[-25, -15])

    ############################
    for patch, color in zip(patches, colors):
        patch.set_color(color * np.ones(3))
        if isinstance(patch, Line2D):
            ax.add_line(patch)
        else:
            patch.set_edgecolor(Black * np.ones(3))
            ax.add_patch(patch)


    arrow = mpatches.FancyArrowPatch((23, -20), (36, -20),color='gray',mutation_scale=10)
    ax.add_patch(arrow)

    arrow = mpatches.FancyArrowPatch((65, -20), (78, -20),color='gray',mutation_scale=10)
    ax.add_patch(arrow)

    arrow = mpatches.FancyArrowPatch((104, -20), (118, -20),color='gray',mutation_scale=10)
    ax.add_patch(arrow)

    arrow = mpatches.FancyArrowPatch((142, -20), (158, -20),color='gray',mutation_scale=10)
    ax.add_patch(arrow)

    arrow = mpatches.FancyArrowPatch((173, -20 ), (195, -20),color='gray',mutation_scale=10)
    ax.add_patch(arrow)

    arrow = mpatches.FancyArrowPatch((212, -20), (232, -20),color='gray',mutation_scale=10)
    ax.add_patch(arrow)

    arrow = mpatches.FancyArrowPatch((252, -20), (273, -20),color='gray',mutation_scale=10)
    ax.add_patch(arrow)


    # ax.add_line(Line2D([170, 200],
    #                       [0, 0],lw=1,ls='--'))
    #
    # ax.add_line(Line2D([170, 220],
    #                       [-30, -40],lw=1,ls='--'))
    #
    # ax.add_line(Line2D([200, 240],
    #                       [0, 0],lw=1,ls='--'))
    #
    # ax.add_line(Line2D([220, 260],
    #                       [-40, -40],lw=1,ls='--'))
    #
    # #
    # ax.add_line(Line2D([240, 280],
    #                       [0, -20],lw=1,ls='--'))
    #
    # ax.add_line(Line2D([260, 280],
    #                       [-40, -20],lw=1,ls='--'))



    plt.tight_layout()
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    fig.set_size_inches(8, 2.5)

    fig_dir = r'E:\huangmachinelearning\draw_convnet-master'
    fig_ext = '.pdf'
    fig.savefig(os.path.join(fig_dir, 'convnet_fig' + fig_ext),
                bbox_inches='tight', pad_inches=0)
