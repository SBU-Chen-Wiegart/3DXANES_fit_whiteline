#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:52:27 2021

@author: karenchen-wiegart
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import h5py
#from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
#from PIL import Image
#import math
import timeit
import glob
import os

# %matplotlib qt
# %matplotlib notebook
# %matplotlib inline
plt.close('all')

# in_dir = '/media/karenchen-wiegart/CHL_14/20201212_FXI_NMC/XANES_Imaging/'
in_dir = '/Volumes/CHL_SBU/MAC2015_backup/20201212_FXI_Co_Mn_XANES/NMC3_standards/'
# img_dir = '/home/karenchen-wiegart/ChenWiegartgroup/Xiaoyin/watershed_20210615/save_png_20210722/'
img_dir = in_dir
fn_list = glob.glob(in_dir + '*_Mn.txt')
fn_list.sort()
i = 0


'''
Saving figures or not, True: saved, False: not saved
'''
saving = True
# saving = False

'''
Plotting style: linweidth, fontsize, colormap (palettable needs be imported)...
'''
labelsize = 24
spinewidth = 3
fontsize = 28
legendsize = 20
fontweight = 'bold'
labelpad = 6
pad = 10
linewidth = 3
tick_size = 6
tick_width = 4
# legend_properties = {'weight':'bold', 'size':24}
markersize= 10
markeredgewidth = 3
transparent = True
# colored_linewidth = 0.01
# palette = pld.get_map('Fall_6', reverse=False)
# cmap = palette.mpl_colormap


'''
Read spectrum from txt files
'''
data = np.loadtxt(fn_list[i])
fn = os.path.basename(fn_list[i])
# data[:,0] /= 1000
data = np.around(data, decimals=6)
# np.savetxt(fn[:-4]+'_keV.txt', data)

# data0 = np.loadtxt(fn_list[0])
# fn0 = os.path.basename(fn_list[0])
# data0 = np.around(data0, decimals=6)

# plt.figure()
# plt.plot(data[:,0], data[:, 1])

'''
Check fitting range: [fit_start, fit_end]
'''
eng_scan = data[:,0]
xanes = data[:,1]

# eng_scan0 = data0[:,0]
# xanes0 = data0[:,1]

max_idx = np.argmax(xanes)
fit_start = max_idx-5
fit_end = max_idx+6


'''
Fit spectrum
'''

x = eng_scan[fit_start:fit_end]
y = xanes[fit_start:fit_end]
# r = np.max(y)/log_tiff_0[-1,i,j]

# weighted arithmetic mean
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))


def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))  # The probability density of the normal distribution


popt,pcov = curve_fit(gaus,x,y,p0=[np.max(y),mean,sigma], maxfev=1000000)
fitted_result = gaus(x,*popt)
perr = np.sqrt(np.diag(pcov))
residulas = y - fitted_result
ss_res = np.sum(residulas**2)
ss_tot = np.sum((y-np.mean(y))**2)
r_2 = 1 - (ss_res / ss_tot)
r2 = f'R\u00b2={r_2:.2f}'
fit_peak = f'fit peak={popt[1]:.2f}'
# r_square[i,j] = r_2


'''
Plot spetrum obtained from txt file
'''
f1, ax1 = plt.subplots(1, 1, figsize = (10, 8))
# Ni4 = r'$\mathbf{Delithiated: Ni^{4+}}$'
# Ni2 = r'$\mathbf{Lithiated: Ni^{2+}}$'
charge = 'Mn in delithiated NMC'
pristine = 'Mn in lithiated NMC'
# ax3.set_yticks((np.arange(ylim[0], ylim[1]+y_inter/10, y_inter)))

if i == 1:
    ax1.plot(eng_scan, xanes, 'b', label=pristine, linewidth=linewidth)
else:
    ax1.plot(eng_scan, xanes, 'r', label=charge, linewidth=linewidth)
x_label1 = 'Energy (eV)'
y_label1 = 'Intensity'

legend_properties = {'weight':'bold', 'size':24}
ax1.legend(fontsize=legendsize, prop=legend_properties)

# xlim = [8325, 8370]
# x_inter = 10
ylim = [-0.07, 1.34]
y_inter = 0.4

# ax1.set_xlim(xlim[0], xlim[1])
# ax1.set_ylim(ylim[0], ylim[1])
# Hide the left and top spines
# ax1.spines['left'].set_visible(False)
# ax1.spines['top'].set_visible(False)
#### ticks
ax1.xaxis.set_ticks_position('bottom')
# ax1.set_xticks(np.arange(xlim[0], xlim[1]+x_inter, x_inter))
ax1.tick_params(axis='x', direction='in', labelsize=labelsize, size=tick_size, width=tick_width)
#ax1.yaxis.set_ticks_position('left')
# ax1.yaxis.tick_right()
# ax1.set_yticks([])
ax1.set_yticks((np.arange(0, ylim[1]+y_inter/10, y_inter)))
ax1.tick_params(axis='y', direction='in', labelsize=labelsize, size=tick_size, width=tick_width)
plt.setp(ax1.get_xticklabels(), fontweight="bold")
plt.setp(ax1.get_yticklabels(), fontweight="bold")
# y_label1 = f'Average Volume / particle ({y_unit})'
# y_label1 = f'Normalized Volume / particle ({y_unit})'
# y_label1 = f'Volume Fraction ({x_unit})'
# x_label1 = f'Distance to Surface ({x_unit})'
# x_label1 = f'Normalized Distance to Surface ({x_unit})'
ax1.set_xlabel(x_label1, labelpad=labelpad, fontsize=fontsize, fontweight=fontweight)
ax1.set_ylabel(y_label1, labelpad=labelpad, fontsize=fontsize, fontweight=fontweight)
ax1.yaxis.set_label_position('left')
ax1.spines["bottom"].set_linewidth(spinewidth)
ax1.spines["right"].set_linewidth(spinewidth)
ax1.spines["top"].set_linewidth(spinewidth)
ax1.spines["left"].set_linewidth(spinewidth)

plt.tight_layout()
plt.show()

# Save plot
if saving == True:       
    imag_name = fn[:-4] + '_01'
    # imag_name = '20210302_BMM_NMC3_pristine_charge_Ni_03'
    plt.savefig(img_dir+imag_name, dpi = 600, transparent=transparent)
else:
    pass


'''
Plot the result of Gaussian fitting
'''
f3, ax3 = plt.subplots(1, 1, figsize = (10, 8))
ax3.plot(eng_scan[fit_start-4:fit_end+5],xanes[fit_start-4:fit_end+5],'b+:',label='data', linewidth=linewidth, markersize=markersize, markeredgewidth = markeredgewidth)
ax3.plot(x,fitted_result,'ro:',label='fit\n'+r2, linewidth=linewidth, markersize=markersize-3, markeredgewidth = markeredgewidth)
ax3.plot([popt[1], popt[1]], [np.min(y),np.max(y)], 'g--', label=fit_peak, linewidth=linewidth)
x_label2 = 'Energy (eV)'
y_label2 = 'Intensity'

legend_properties = {'weight':'bold', 'size':38}
ax3.legend(fontsize=legendsize, prop=legend_properties)



if i == 1:
    xlim = [6558, 6563]
    x_inter = 2
    ylim = [1.07, 1.19]
    y_inter = 0.04
else:
    xlim = [6559, 6564.5]
    x_inter = 2
    ylim = [1.10, 1.30]
    y_inter = 0.04    


# ax3.set_ylim(ylim[0], ylim[1])
# Hide the left and top spines
# ax3.spines['left'].set_visible(False)
# ax3.spines['top'].set_visible(False)
#### ticks
ax3.xaxis.set_ticks_position('bottom')
ax3.set_xticks((np.arange(xlim[0], xlim[1]+x_inter/10, x_inter)))
ax3.set_xlim(xlim[0]-0.2, xlim[1]+0.2)
ax3.tick_params(axis='x', direction='in', labelsize=labelsize, size=tick_size, width=tick_width)
plt.setp(ax3.get_xticklabels(), fontweight="bold")
plt.setp(ax3.get_yticklabels(), fontweight="bold")
#ax3.yaxis.set_ticks_position('left')
# ax3.yaxis.tick_right()
# ax3.set_yticks([])
ax3.set_yticks((np.arange(ylim[0], ylim[1]+y_inter/10, y_inter)))
ax3.set_ylim(ylim[0]-0.005, ylim[1]+0.002)
ax3.tick_params(axis='y', direction='in', labelsize=labelsize, size=tick_size, width=tick_width)
# y_label2 = f'Average Volume / particle ({y_unit})'
# y_label2 = f'Normalized Volume / particle ({y_unit})'
# y_label2 = f'Volume Fraction ({x_unit})'
# x_label2 = f'Distance to Surface ({x_unit})'
# x_label2 = f'Normalized Distance to Surface ({x_unit})'
ax3.set_xlabel(x_label2, labelpad=labelpad, fontsize=fontsize, fontweight=fontweight)
ax3.set_ylabel(y_label2, labelpad=labelpad, fontsize=fontsize, fontweight=fontweight)
ax3.yaxis.set_label_position('left')
ax3.spines["bottom"].set_linewidth(spinewidth)
ax3.spines["right"].set_linewidth(spinewidth)
ax3.spines["top"].set_linewidth(spinewidth)
ax3.spines["left"].set_linewidth(spinewidth)

plt.tight_layout()
plt.show()


# Save plot
if saving == True:
    imag_name = f'{fn[:-4]}_fit_02'
    plt.savefig(img_dir+imag_name, dpi = 600, transparent=transparent)
else:
    pass



