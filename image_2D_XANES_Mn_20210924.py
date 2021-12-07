#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 02:12:07 2021

@author: chenghung
"""

import matplotlib.pyplot as plt
# import tomopy
import numpy as np
#from PIL import Image 
from skimage import io
import palettable.colorbrewer.diverging as pld
import os

# %matplotlib qt
# %matplotlib notebook
# %matplotlib inline

plt.close('all')

palette = pld.PuOr_10_r
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

fn = fn_x0['C200_Mn']
vmin = 6.558
vmax = 6.563

x0_fit = io.imread(fn)
x0_fit[x0_fit < vmin] = 0
x0_fit[x0_fit > vmax] = 0
x0_fit[x0_fit == 0] = np.nan
plt.figure()
plt.imshow(x0_fit, interpolation= None, vmin=vmin, vmax=vmax, cmap = cmap)
# plt.imshow(x0_fit, interpolation= None, cmap = cmap)
plt.xticks(ticks=[], labels=[])
plt.yticks(ticks=[], labels=[])
cbar = plt.colorbar()
cbar.ax.set_yticklabels(np.arange(int(vmin*1000), int(vmax*1000+1), 1), fontsize=12, weight='bold')

scan_id = int(fn[-10:-5])
out_png_dir = os.path.dirname(fn)

imag_name = f'Gaussain_whiteline_{scan_id}_bin4_03_2.png'
#out_png_dir = '/home/xf18id/Desktop/users/2019Q3/KAREN_Proposal_305052/41544_41651/fitted_png/'
out_png = out_png_dir + '/' + imag_name
plt.savefig(out_png, dpi = 600, transparent=True)

'''
ax = plt.gca()
ax.set_facecolor('k')
# cbar = plt.colorbar(ticks=np.arange(start_eng, end_eng+0.1, 0.01))
cbar = plt.colorbar()
cbar.set_label('White Line (keV)', labelpad = 10, fontsize=15, fontweight='bold')
cbar.ax.tick_params(labelsize=12)
#cbar.ax.set_yticklabels([6.558, 6.559, 6.56])
#imag_name = inputdir + 'Gaussain_' + slice_num
imag_name = f'Gaussain_whiteline_{scan_id}_bin2.png'
#out_png_dir = '/home/xf18id/Desktop/users/2019Q3/KAREN_Proposal_305052/41544_41651/fitted_png/'
out_png = out_tiff_dir + imag_name
plt.savefig(out_png, dpi = 300, transparent=True)
# ps = 'Fitting of slice ' + "%03d" %slice_num + ' is done!'
# print(ps)

stop_time = timeit.default_timer()
elapse_time = stop_time - start_time
print(f'Elapsed time for fitting slice {scan_id}: {elapse_time: .2f} seconds.') 
'''

plt.show()
