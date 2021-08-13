#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:59:41 2019

@author: chenghung
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
#import h5py
#from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
#from PIL import Image
#import math

# %matplotlib qt
# %matplotlib notebook
# %matplotlib inline


plt.close('all')
#inputdir = '/Users/chenghung/Desktop/FXI'

'''
Read 2D XANES from tiff and enegy points from h5
'''
scanID=15745
condition = 'B91_pristine'   #Sample condition
tiff_file = 'xanes_scan2_id_'+str(scanID)+'_aligned.tiff'
#h5_file = 'xanes_scan2_id_'+str(scanID)+'.h5'    # Note: Foil samples used 'xanes_scan_id', other smaples used 'xanes_scan2_id_'
#ratio_file = str(scanID)+'_ratio.tiff'
#ratio_file = str(scanID)+'_0327_ratio.tiff'
mask_file = str(scanID)+'_mask.tiff'

tiff = io.imread(tiff_file)
#ratio = io.imread(ratio_file)
mask = io.imread(mask_file)
#print(tiff.shape)
#print(mask.shape)

#h5 = h5py.File(h5_file, 'r')
#print(list(h5.keys()))
#eng_scan=np.array(h5['X_eng'])
#print(eng_scan)

eng_scan = np.array([6.457 , 6.467 , 6.477 , 6.487 , 6.497 , 6.507 , 6.517 , 6.527 ,
       6.5355, 6.536 , 6.5365, 6.537 , 6.5375, 6.538 , 6.5385, 6.539 ,
       6.5395, 6.54  , 6.5405, 6.541 , 6.5415, 6.542 , 6.5425, 6.543 ,
       6.5435, 6.544 , 6.5445, 6.545 , 6.5455, 6.546 , 6.5465, 6.547 ,
       6.5475, 6.548 , 6.5485, 6.549 , 6.5495, 6.55  , 6.5505, 6.551 ,
       6.5515, 6.552 , 6.5525, 6.553 , 6.5535, 6.554 , 6.5545, 6.555 ,
       6.5555, 6.556 , 6.5565, 6.557 , 6.5575, 6.558 , 6.5585, 6.559 ,
       6.5595, 6.56  , 6.5605, 6.561 , 6.5615, 6.562 , 6.5625, 6.563 ,
       6.5635, 6.564 , 6.5645, 6.565 , 6.5655, 6.566 , 6.5665, 6.567 ,
       6.5675, 6.568 , 6.5685, 6.569 , 6.5695, 6.57  , 6.572 , 6.579 ,
       6.586 , 6.593 , 6.6   , 6.607 , 6.614 , 6.629 , 6.654 , 6.679 ,
       6.704 , 6.729 , 6.754 , 6.779 , 6.804 ])

'''
Take negative log to get x-ray attenuation
'''
log_tiff = -np.log(tiff)


'''
Set all non-finite values (infinity and NaN) to 0!!
'''
log_tiff_0 = log_tiff[:]
finite = np.isfinite(log_tiff_0)
non_finite = np.invert(finite)
log_tiff_0[non_finite] = 0    
#io.imsave('xanes_scan2_id_'+str(scanID)+'_aligned_log.tiff' , log_tiff_0)
#io.imsave('cell_03_aligned_02.tiff' , insitu)

'''
Display a single fram of image before -log if desired:
'''
showframe = 53 #select which frame to show, where engergy == 6.558 kV
plt.figure('xanes_scan2_id_'+str(scanID)+'_'+str(eng_scan[showframe]))
plt.imshow(log_tiff_0[showframe], interpolation= None, vmin=0, vmax=2) #interpolation can be 'nearest' also. 
cbar = plt.colorbar()
#cbar.set_label('X-ray Absorption', fontsize=14)


'''
Plot spectrum from single pixel
'''
x_pixel = 540
y_pixel = 680
plt.figure()
plt.plot(eng_scan, log_tiff_0[:,x_pixel,y_pixel], 'b')
imag_name = 'Individual_XANES' 
# plt.savefig(imag_name, dpi = 300)

'''
Gaussian fitting for a singal pixel
'''
fit_start = 47
fit_end = 64
x = eng_scan[fit_start:fit_end]
y = log_tiff_0[fit_start:fit_end ,x_pixel ,y_pixel]

# weighted arithmetic mean
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))


def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))  # The probability density of the normal distribution


popt,pcov = curve_fit(gaus,x,y,p0=[np.max(y),mean,sigma])
fitted_result = gaus(x,*popt)

plt.figure()
plt.plot(x,y,'b+:',label='data')
plt.plot(x,fitted_result,'ro:',label='fit')
plt.legend()
#plt.title('Fig. 3 - Fit for Time Constant')
#plt.xlabel('Time (s)')
#plt.ylabel('Voltage (V)')
#plt.show()
#imag_name = 'Gaussain_individual_fitting' 
#plt.savefig(imag_name, dpi = 300)


'''
Gaussian fitting for an ROI
'''
#x_start = 630
#x_end = 630+80
#y_start = 720
#y_end = 720+20
#mean_roi = np.mean(np.mean(log_tiff_0[ : , x_start:x_end , y_start:y_end], axis=1), axis=1)
#
#plt.figure()
#plt.plot(eng_scan, mean_roi, 'g')
#
#fit_start = 47
#fit_end = 64
#x = eng_scan[fit_start:fit_end]
#y = mean_roi[fit_start:fit_end]
#
## weighted arithmetic mean
#mean = sum(x * y) / sum(y)
#sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
#
#
#def gaus(x,a,x0,sigma):
#    return a*exp(-(x-x0)**2/(2*sigma**2))  # The probability density of the normal distribution
#
#
#popt,pcov = curve_fit(gaus,x,y,p0=[np.max(y),mean,sigma])
#fitted_result = gaus(x,*popt)
#
#plt.figure()
#plt.plot(x,y,'b+:',label='data')
#plt.plot(x,fitted_result,'ro:',label='fit')
#plt.legend()



'''
Plot spectrum in the first derivative
'''
##d_sum_roi = np.diff(sum_roi)/np.diff(eng_scan)
##plt.plot(d_sum_roi, 'b')
#gra_tiff = np.gradient(log_tiff_0, eng_scan, axis = 0)
#plt.figure()
#plt.plot(eng_scan, g_roi, 'g')



'''
Gaussian fitting for Field of view
'''

fit_start = 47
fit_end = 64
x = eng_scan[fit_start:fit_end]
y = log_tiff_0[fit_start:fit_end ,: ,:]

# weighted arithmetic mean
Y = (y.T*x).T
mean = np.sum(Y, axis = 0) / np.sum(y, axis = 0)
X = np.zeros(y.shape)
for i in range(len(x)):
    X[i,:,:] = x[i]
XY = y * (X - mean)**2
sigma = np.sqrt(np.sum(XY, dtype=np.float64, axis = 0) / np.sum(y, dtype=np.float64, axis = 0))

mean = np.nan_to_num(mean)
sigma = np.nan_to_num(sigma)

#y_mask = np.multiply(y, mask)
#mean_mask = np.multiply(mean, mask)
#sigma_mean = np.multiply(sigma, mask)

def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))  # The probability density of the normal distribution

fitted_result = np.zeros ((y.shape[1], y.shape[2]))

for i in range(y.shape[1]):
    for j in range(y.shape[2]):
        if mask[i,j] == 0:
            pass
    
        else:
            popt,pcov = curve_fit(gaus,x,y[:,i,j],p0=[np.max(y[:,i,j]),mean[i,j],sigma[i,j]], maxfev=100000)
            fitted_single = gaus(x,*popt)
            fitted_result[i,j] = x[np.argmax(fitted_single)]

fitted_result = fitted_result[200:900, 300:1000]
fitted_result[fitted_result == 0] = np.nan
plt.figure()
plt.imshow(fitted_result, interpolation= None, vmin=6.555, vmax=6.560, cmap = "Spectral")
plt.xticks(ticks=[], labels=[])
plt.yticks(ticks=[], labels=[])
ax = plt.gca()
ax.set_facecolor('k')
cbar = plt.colorbar()
cbar.set_label('White Line (keV)', labelpad = 10, fontsize=15, fontweight='bold')
cbar.ax.tick_params(labelsize=12)
imag_name = 'Gaussain_' +condition+ '_' + str(scanID) + '.tiff'
io.imsave(imag_name, fitted_result) 
# plt.savefig(imag_name, dpi = 300)


'''
Select ROI
'''
#color = 'r'
##ROI_i
#x_0 =  630-300
#x_1 =  x_0 + 80
#y_0 = 300-200
#y_1 = y_0 + 20
#plt.plot( [x_0, x_1], [y_0, y_0], color)
#plt.plot( [x_0, x_1], [y_1, y_1], color)
#plt.plot( [x_0, x_0], [y_0, y_1], color)
#plt.plot( [x_1, x_1], [y_0, y_1], color)
#
###ROI_ii
#x_0 =  630-300
#x_1 =  x_0 + 80
#y_0 = 530-200
#y_1 = y_0 + 20
#plt.plot( [x_0, x_1], [y_0, y_0], color)
#plt.plot( [x_0, x_1], [y_1, y_1], color)
#plt.plot( [x_0, x_0], [y_0, y_1], color)
#plt.plot( [x_1, x_1], [y_0, y_1], color)
##
###ROI_iii
#x_0 =  630-300
#x_1 =  x_0 + 80
#y_0 = 720-200
#y_1 = y_0 + 20
#plt.plot( [x_0, x_1], [y_0, y_0], color)
#plt.plot( [x_0, x_1], [y_1, y_1], color)
#plt.plot( [x_0, x_0], [y_0, y_1], color)
#plt.plot( [x_1, x_1], [y_0, y_1], color)
##
###ROI_iv
##x_0 =  750-300
##x_1 =  x_0 + 30
##y_0 = 780-200
##y_1 = y_0 + 30
##plt.plot( [x_0, x_1], [y_0, y_0], color)
##plt.plot( [x_0, x_1], [y_1, y_1], color)
##plt.plot( [x_0, x_0], [y_0, y_1], color)
##plt.plot( [x_1, x_1], [y_0, y_1], color)
#
#plt.show()
#
#imag_name = 'max_eng_mask_FIB_' + str(scanID) + '_04'
##plt.savefig(imag_name, dpi = 300)

'''
Plot spectrum from above selected ROI
'''
#x_start = x_0 + 300
#x_end = x_start + 80
#y_start = y_0 + 200
#y_end = y_start + 20
#mean_roi = np.mean(np.mean(log_tiff_0[ : , x_start:x_end , y_start:y_end], axis=1), axis=1)
#
#plt.figure()
#plt.plot(eng_scan, mean_roi, 'g')



'''
Try to find the highest/maxium intensity of each spectrum
'''
# max_index = np.argmax(log_tiff_0, axis = 0)
# max_index_mask = np.multiply(max_index, mask)
# max_eng = eng_scan[max_index]
# max_eng_mask = np.multiply(max_eng, mask)

# max_eng_mask_01 = max_eng_mask[200:900, 300:1000]
# max_eng_mask_01[max_eng_mask_01 == 0] = np.nan

# plt.figure()
# plt.imshow(max_eng_mask_01, vmin = 6.555, vmax = 6.560, cmap = "Spectral")
# plt.xticks(ticks=[], labels=[])
# plt.yticks(ticks=[], labels=[])
# ax = plt.gca()
# ax.set_facecolor('k')
# cbar = plt.colorbar()
# cbar.set_label('White Line (keV)', labelpad = 10, fontsize=15, fontweight='bold')
# cbar.ax.tick_params(labelsize=12)
# imag_name = 'Max_' +condition+ '_' + str(scanID)
# plt.savefig(imag_name, dpi = 300)