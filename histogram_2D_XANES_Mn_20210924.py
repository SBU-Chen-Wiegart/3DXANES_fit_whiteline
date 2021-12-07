# -*- coding: utf-8 -*-
"""
Created on Tue May 11 16:32:55 2021

@author: Chen-Wiegart
"""

import matplotlib.pyplot as plt
# import tomopy
import numpy as np
#from PIL import Image 
from skimage import io
import palettable.colorbrewer.qualitative as pld
# from matplotlib.ticker import PercentFormatter
# import os
# import time



# %matplotlib qt
# %matplotlib notebook
# %matplotlib inline

plt.close('all')

palette = pld.Set1_4
cmap = palette.mpl_colormap
transparency = 0.3
labelsize = 20
spinewidth = 3
fontsize = 32
legendsize = 20
fontweight = 'bold'
labelpad = 6
pad = 10
linewidth = 2.5
tick_size = 6
tick_width = 4
legend_properties = {'weight':'bold', 'size':22}

# sample = ['S0', 'C0', 'S200', 'C200']
# spl_idx = 0
in_dir = '/Volumes/CHL_SBU/MAC2015_backup/20201212_FXI_Co_Mn_XANES/'

C200_Co = '79929_C200_Co_loc1_side/fit_white_bin2_03/x0_fit_79929.tiff'
C200_Mn = '79930_C200_Mn_loc1_side/fit_white_bin2_02/x0_fit_79930.tiff'
S200_Co = '80483_S200_Co_loc1_side/fit_white_bin2_03/x0_fit_80483.tiff'
S200_Mn = '80484_S200_Mn_loc1_side/fit_white_bin2_03/x0_fit_80484.tiff'

fn_x0 = {'C200_Co' : in_dir + C200_Co,
         'C200_Mn' : in_dir + C200_Mn,
         'S200_Co' : in_dir + S200_Co,
         'S200_Mn' : in_dir + S200_Mn
         }


'''
Plot histogram
'''

# fn_list = [fn_x0['S200_cross'], fn_x0['S200_top']]
# label_list = ['SP 200 cycles cross-section', 'SP 200 cycles top view']

fn_list = [fn_x0['C200_Mn'], fn_x0['S200_Mn']]
label_list = ['CNT 200 cycles - Mn', 'SP 200 cycles - Mn']



# color_idx = np.linspace(0, 1, len(fn_list))
color_idx = np.linspace(0, 1, 4)
# color_list = range(len(color_idx))  # for S0, C0, S200, C200
# color_list = [0, 2]  # for S0 and S200
color_list = [1, 3]  # for C0 and C200
# color_list = [0, 0, 0]  # for S0
# color_list = [2, 2, 2]  # for S200
# color_list = [1, 1, 1]  # for C0
# color_list = [3, 3, 3]  # for C200
# transparency = [0.3, 0.5, 0.7]
f1, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 8), gridspec_kw={'width_ratios': [1,1]})
# plt.title('Size of 3D Mean filter: 15 pixels', fontsize=fontsize, fontweight=fontweight, pad=pad, x=0)
# f1, (ax1, ax2) = plt.subplots(1, 2)

# for i in range(len(fn_list)):
for i, j in zip(range(len(fn_list)), color_list):
    
    img = io.imread(fn_list[i])*1000
    # print(f'{label_list[i]}: {img.shape}')  
    # img = io.imread(fn_list[i])[:, 10:75, :]
    m = 6558
    M = 6563 #img.max()
    img[img==0] = np.nan
    flat = img.flatten()
    hist, bins = np.histogram(flat, bins = 256, range=(m, M))
    s = np.sum(hist)
    ax1.hist(flat, bins=256, range=(m, M), color=cmap(color_idx[j]), alpha=transparency, label=label_list[i])
    ax2.plot(bins[:-1], hist/s, color=cmap(color_idx[j]), alpha=transparency, linewidth=linewidth)
    ax2.fill_between(bins[:-1], hist/s, color=cmap(color_idx[j]), alpha=transparency, label=label_list[i])
    # ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    '''
    img1 = io.imread(fn_list[i])[:, 10:80, :]
    img2 = io.imread(fn_list[i])[:, -80:-10, :]
    M = 0.03 #img.max()
    img1[img1==0] = np.nan
    img2[img2==0] = np.nan
    flat1 = img1.flatten()
    flat2 = img2.flatten()
    hist1, bins1 = np.histogram(flat1, bins = 256, range=(0, M))
    hist2, bins2 = np.histogram(flat2, bins = 256, range=(0, M))
    s1 = np.sum(hist1)
    s2 = np.sum(hist2)
    ax1.hist(flat1, bins=256, range=(0, M), color=cmap(color_idx[j]), alpha=transparency, label=label_list[i]+ ' Surface')
    ax1.hist(flat2, bins=256, range=(0, M), color=cmap(color_idx[j]), alpha=transparency+0.3, label=label_list[i]+ ' Collector')
    ax2.plot(bins1[:-1], hist1/s1, color=cmap(color_idx[j]), alpha=transparency)
    ax2.fill_between(bins1[:-1], hist1/s1, color=cmap(color_idx[j]), alpha=transparency, label=label_list[i]+ ' Surface')
    ax2.plot(bins2[:-1], hist2/s2, color=cmap(color_idx[j]), alpha=transparency+0.3)
    ax2.fill_between(bins2[:-1], hist2/s2, color=cmap(color_idx[j]), alpha=transparency+0.3, label=label_list[i]+ ' Collector')
    '''


ax1.legend(fontsize=legendsize) 
ax2.legend(fontsize=legendsize, prop=legend_properties)

xlim = [0, 0.03]
x_inter = 0.01
ylim = [-0.0015, 0.143]
y_inter = 0.02

# ax1.set_xlim(xlim[0], xlim[1])
# ax1.set_ylim(ylim[0], ylim[1])
# Hide the left and top spines
# ax1.spines['left'].set_visible(False)
# ax1.spines['top'].set_visible(False)
#### ticks
ax1.xaxis.set_ticks_position('bottom')
# ax1.set_xticks(np.arange(xlim[0], xlim[1]+x_inter, x_inter))
ax1.tick_params(axis='x', direction='in', labelsize=labelsize)
#ax1.yaxis.set_ticks_position('left')
# ax1.yaxis.tick_right()
# ax1.set_yticks([])
ax1.tick_params(axis='y', direction='in', labelsize=labelsize)
plt.setp(ax1.get_xticklabels(), fontweight="bold")
plt.setp(ax1.get_yticklabels(), fontweight="bold")
y_label1 = 'Counts'
x_label1 = 'White Line (eV)'
ax1.set_xlabel(x_label1, labelpad=labelpad, fontsize=fontsize, fontweight=fontweight)
ax1.set_ylabel(y_label1, labelpad=labelpad, fontsize=fontsize, fontweight=fontweight)
ax1.yaxis.set_label_position('left')
# ax1.spines["bottom"].set_linewidth(spinewidth)
# ax1.spines["right"].set_linewidth(spinewidth)


# ax2.set_xlim(xlim[0], xlim[1])
ax2.set_ylim(ylim[0], ylim[1])
# Hide the left and top spines
# ax2.spines['left'].set_visible(False)
# ax2.spines['top'].set_visible(False)
#### ticks
ax2.xaxis.set_ticks_position('bottom')
# ax2.set_xticks((np.arange(xlim[0], xlim[1]+x_inter, x_inter)))
ax2.yaxis.tick_right()
ax2.tick_params(axis='x', direction='in', labelsize=labelsize, size=tick_size, width=tick_width)
plt.setp(ax2.get_xticklabels(), fontweight="bold")
plt.setp(ax2.get_yticklabels(), fontweight="bold")
#ax2.yaxis.set_ticks_position('left')
# ax2.set_yticks([])
ax2.set_yticks((np.arange(0, ylim[1]+y_inter/10, y_inter)))
ax2.tick_params(axis='y', direction='in', labelsize=labelsize, size=tick_size, width=tick_width)
y_label2 = 'Normalized Counts'
x_label2 = 'White Line (eV)'
ax2.set_xlabel(x_label2, labelpad=labelpad, fontsize=fontsize, fontweight=fontweight)
ax2.set_ylabel(y_label2, labelpad=labelpad, fontsize=fontsize, fontweight=fontweight)
ax2.yaxis.set_label_position('right')
ax2.spines["bottom"].set_linewidth(spinewidth)
ax2.spines["right"].set_linewidth(spinewidth)
ax2.spines["top"].set_linewidth(spinewidth)
ax2.spines["left"].set_linewidth(spinewidth)

plt.tight_layout()
plt.show()

img_fn = 'his_2D_XANES_CNT_Mn_02_1'
plt.savefig(in_dir+img_fn, dpi=600, transparent=True)

