#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:46:50 2020

@author: chenghung
"""

import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
#from skimage import io
from scipy.optimize import curve_fit
#from scipy import asarray as ar,exp
from scipy.signal import find_peaks

# %matplotlib qt
# %matplotlib notebook
# %matplotlib inline
plt.close('all')


'''
Define Gaussian function
'''
def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))  # The probability density of the normal distribution

'''
Saving figures or not, True: saved, False: not saved
'''
# saving = True
saving = False

'''
Plotting style: linweidth, fontsize, colormap (palettable needs be imported)...
'''
spinewidth = 2
linewidth = 2
labelpad = 12
fontsize = 24
labelsize = 16
fontweight = 'bold'
spinewidth = 2
transparent = True
# colored_linewidth = 0.01
# palette = pld.get_map('Fall_6', reverse=False)
# cmap = palette.mpl_colormap

'''
Read spectrum from txt files
'''
inputdir = '/media/karenchen-wiegart/hard_disk/CHLin/20210321_FXI/92266_92267_SP1_2D_XANES/91490_Ni/'
# inputdir = 'J:\\20210321_FXI_3D_XANES_91720_91829_B2_SP200\\91490_Ni\\'
#inputdir = '/Users/chenghung/Desktop/ss/'
save_png = inputdir
#save_png = '/Users/chenghung/Desktop/ss/'
#file_list = glob.glob(inputdir + '*.txt')
#file_list.sort()
fn = 'Ni_91490_spec.txt'
energy=[]
inten=[]


   
# Turn data in txt into list
with open(inputdir+fn) as txt:
    infile = txt.readlines()
    infile2 = infile[1:]   #skip header rows
    for data in infile2:  #each 'data' is a line
        data = data.strip('\n')  #removing the newline character
        data = data.split(' ')  #using space characters to seperate the string into a list of 2 strings
        data = np.array(data)    #converting list to numpy array of strings
        data = data.astype('float')  #converting numpy array strings into numpy array floats
        #print(data)
        energy.append(data[0])
        inten.append(data[1])
        #all_data.append([theta])
        #all_data.append([inten])
    txt.close()
energy = np.array(energy)
inten = np.array(inten)


# Plot spetrum obtained from txt file
f1, ax1 = plt.subplots(1, 1, figsize = (12, 8))
ax1.plot(energy, inten, 'b', label=fn, linewidth=linewidth)
x_label = 'Energy (keV)'
y_label = 'Intensity'
ax1.set_title(f'{fn[:-4]}: Ni foil', fontsize=fontsize, fontweight=fontweight)
ax1.set_xlabel(x_label, labelpad=labelpad, fontsize=fontsize, fontweight=fontweight)
ax1.set_ylabel(y_label, labelpad=labelpad, fontsize=fontsize, fontweight=fontweight)
ax1.tick_params(axis='x', direction='in', labelsize=labelsize)
ax1.tick_params(axis='y', direction='in', labelsize=labelsize)
plt.setp(ax1.get_xticklabels(), fontweight=fontweight)
plt.setp(ax1.get_yticklabels(), fontweight=fontweight)
# plt.xlim(8.32, 8.37)
# x_max = np.around(np.amax(energy)+0.02, decimals = 2)
# x_min = np.around(np.amin(energy)-0.01, decimals = 2)
# plt.xticks(np.arange(x_min, x_max, 0.05))        
ax1.legend(fontsize=fontsize)
# Save plot
if saving == True:       
    imag_name = fn[:-4]
    plt.savefig(save_png+imag_name, dpi = 300, transparent=transparent)
else:
    pass



### If data is Mn or Ni foil, fit the 1st derivative.

# Caculate the 1st derivative of the spetrum for Mn or Ni foil and find the largest peak
grad_int = np.gradient(inten, energy, axis = 0)
peaks, _ = find_peaks(grad_int)
peak_max = np.max(grad_int[peaks])
# peak_max_ind = np.argwhere(grad_int == peak_max)
peak_70 = np.argwhere(grad_int[peaks] > peak_max*0.7)
### Select peak order
order = 1
peak_70_idx = np.argwhere(grad_int == grad_int[peaks][peak_70][order][0])

# Plot derivative and peak position
f2, ax2 = plt.subplots(1, 1, figsize = (12, 8))
ax2.plot(energy, grad_int, 'b--+', label='1st der', linewidth=linewidth, markersize= 10, markeredgewidth = 2,)
x_label = 'Energy (keV)'
y_label = 'Intensity'
ax2.set_title(f'{fn[:-4]}: 1st Derivative', fontsize=fontsize, fontweight=fontweight)
ax2.set_xlabel(x_label, labelpad=labelpad, fontsize=fontsize, fontweight=fontweight)
ax2.set_ylabel(y_label, labelpad=labelpad, fontsize=fontsize, fontweight=fontweight)
ax2.tick_params(axis='x', direction='in', labelsize=labelsize)
ax2.tick_params(axis='y', direction='in', labelsize=labelsize)
plt.setp(ax2.get_xticklabels(), fontweight=fontweight)
plt.setp(ax2.get_yticklabels(), fontweight=fontweight)
ax2.plot(energy[peak_70_idx[0][0]], grad_int[peak_70_idx[0][0]], "rx", markersize= 15, markeredgewidth = 5,
         label=f'first peak\n={energy[peak_70_idx][0][0]:.4f} (keV)')
ax2.legend(fontsize=fontsize)
# Save plot
if saving == True:
    imag_name = f'{fn[:-4]}_der_01'
    plt.savefig(save_png+imag_name, dpi = 300)
else:
    pass
#plt.close()
    
# Find the range of Gaussian fitting 
fit_start = peak_70_idx[0][0]-12
fit_end = peak_70_idx[0][0]+12
x = energy[fit_start:fit_end]
y = grad_int[fit_start:fit_end]

# weighted arithmetic mean
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

# Gaussian fitting
popt,pcov = curve_fit(gaus,x,y,p0=[np.max(y),mean,sigma], maxfev=10000)
fitted_result = gaus(x, *popt)
perr = np.sqrt(np.diag(pcov))
residulas = y - fitted_result
ss_res = np.sum(residulas**2)
ss_tot = np.sum((y-np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)
r2 = f'R\u00b2={r_squared:.2f}'
#r2 = r'$R^{2}$'

# Plot the result of Gaussian fitting
f3, ax3 = plt.subplots(1, 1, figsize = (12, 8))
# ax3.plot(energy, grad_int, 'b--+', label='1st der', linewidth=linewidth, markersize= 10, markeredgewidth = 2,)
ax3.plot(x,y,'b+:',label='data', markersize= 6, markeredgewidth = 0.5)
ax3.plot(x,fitted_result,'ro:',label='fit\n'+r2, markersize= 6, markeredgewidth = 0.5)
ax2.plot(x,fitted_result,'ro:',label='fit\n'+r2, markersize= 6, markeredgewidth = 0.5)
ax3.plot([popt[1], popt[1]], [np.min(y),np.max(y)], 'g--', label='fit peak', markersize= 6, markeredgewidth = 0.5)
x_label = 'Energy (keV)'
y_label = 'Intensity'
ax3.set_title(f'{fn[:-4]}: 1st Der, fit peak = {popt[1]: .4f}\u00B1{perr[1]: .4f}', fontsize=fontsize, fontweight=fontweight)
ax3.set_xlabel(x_label, labelpad=labelpad, fontsize=fontsize, fontweight=fontweight)
ax3.set_ylabel(y_label, labelpad=labelpad, fontsize=fontsize, fontweight=fontweight)
ax3.tick_params(axis='x', direction='in', labelsize=labelsize)
ax3.tick_params(axis='y', direction='in', labelsize=labelsize)
plt.setp(ax3.get_xticklabels(), fontweight=fontweight)
plt.setp(ax3.get_yticklabels(), fontweight=fontweight)
ax3.legend(fontsize=fontsize)
x_max = np.around(np.amax(x)+0.0005, decimals = 3)
x_min = np.around(np.amin(x)-0.0005, decimals = 3)
ax3.set_xticks(np.arange(x_min, x_max, 0.001))


# Save plot
if saving == True:
    imag_name = f'{fn[:-4]}_fitting_01'
    plt.savefig(save_png+imag_name, dpi = 300)
else:
    pass
#plt.close()        

print(f'Fitting 1st Deri of {fn[:-4]} is done! Peak = {popt[1]:.6f} eV')
