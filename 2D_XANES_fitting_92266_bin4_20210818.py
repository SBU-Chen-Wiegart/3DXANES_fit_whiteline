# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:50:52 2019

@author: chenlin
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import h5py
#from scipy.signal import find_peaks
from scipy.optimize import curve_fit
# from scipy import asarray as ar,exp
#from PIL import Image
#import math
import timeit
import copy

# %matplotlib qt
# %matplotlib notebook
# %matplotlib inline
plt.close('all')

start_time = timeit.default_timer()


h5_dir = '/media/karenchen-wiegart/hard_disk/CHLin/20210321_FXI/92266_92267_SP1_2D_XANES/92266_SP1_side/'
xanes_dir = '/media/karenchen-wiegart/hard_disk/CHLin/20210321_FXI/92266_92267_SP1_2D_XANES/92266_SP1_side/'
# in_tiff_dir = xanes_dir + 'fit_white_bin2_07/'
out_tiff_dir = xanes_dir + 'fit_white_bin4/'
eng_shift = -0.0004 #keV
maxfev=1000000
popt2 = 0.009  ### Minimun of the Variance in Gaussian function
# popt2_pre = 0.008

'''
Read Energy list from pre-defined list
'''
# NiK=8.333
# Ni_eng_list = list(np.arange(NiK-0.082, NiK-0.022,0.005)) + list(np.arange(NiK-0.021, NiK+0.08, 0.001)) + list(np.arange(NiK+0.09, NiK+0.288, 0.01))
# Ni_eng_list_3 =  list(np.arange(NiK-0.033, NiK-0.004,0.01)) + list(np.arange(NiK-0.004, NiK+0.008, 0.0005)) + list(np.arange(NiK+0.008, NiK+0.022, 0.00025)) + list(np.arange(NiK+0.022, NiK+0.08, 0.005)) + list(np.arange(NiK+0.09, NiK +0.22, 0.01))
# eng = np.array(Ni_eng_list_3) + eng_shift

'''
Read Energy list from h5 file
'''
scan_id = 92266
fn = f'xanes_scan2_id_{scan_id}.h5'
h5_file = h5_dir + fn
h5 = h5py.File(h5_file, 'r')
# # print(list(h5.keys()))
eng = np.array(h5['X_eng']) + eng_shift
# print(eng)
# img_bkg = np.array(h5['img_bkg'])
# img_dark = np.array(h5['img_dark'])
# img_xanes = np.array(h5['img_xanes'])
h5.close()
# xanes = -np.log((img_xanes-img_dark)/(img_bkg-img_dark))


'''
Read aligned, normalized xanes from tiff file
'''
fn = f'xanes_2d_{scan_id}_bin4_shift_log_norm.tiff'
xanes = io.imread(xanes_dir + fn)
# plt.figure()
# plt.imshow(xanes[81])
# plt.colorbar()

'''
Plot spectrum from a single pixel
'''
x_pixel = 48 #73 #173 #226 #48 #291 #166 #402
y_pixel = 63 #602 #572 #361 #626 #270 #577 #540
plt.figure()
plt.plot(eng, xanes[:,x_pixel,y_pixel], 'b')
#imag_name = 'Single_XANES_' + condition + '_(' + str(x_pixel) + ', ' + str(y_pixel) + ')' 
#plt.savefig(imag_name, dpi = 300)

'''    
Set a threshold value with averaging insterest post-edge
'''    
eng_scan = np.around(eng, 6)
index_post = np.where(eng_scan > 8.4)
index_pre = np.where(eng_scan < 8.33)
post_mean = np.mean(xanes[index_post], axis = 0)
pre_mean = np.mean(xanes[index_pre], axis = 0)
threshold_post = [0.5, 2]
threshold_pre = [0, 0.25]
mask = ((threshold_post[0] <= post_mean) * (post_mean <= threshold_post[1]) * (threshold_pre[0] <= pre_mean) * (pre_mean <= threshold_pre[1]))
#mask = mask.astype(int)
#plt.figure()
#plt.imshow(mask, interpolation= None)
#fn = inputdir + 'mask_' + f'{slice_num:04d}' + '.tiff'
#io.imsave(fn, mask)

'''
Fit spectrum from a single pixel
'''
#### Identify max value aorund the white line and give a fitting range [max_idx-start : max_idx+end]
white_0 = 50
white_1 = 90
start = 14 #+ 2
end = 12 #- 2
# max_idx = np.argmax(xanes[white_0:white_1, x_pixel, y_pixel]) + white_0
# fit_start = max_idx - start
# fit_end = max_idx + end

### Directly give the range of white line peak 
# start_eng = 8.35175
# end_eng = 8.35575
# fit_start = np.argwhere(eng_scan == start_eng)[0][0]-5
# fit_end = np.argwhere(eng_scan == end_eng)[0][0]+2
# diff = eng_scan[fit_end] - eng_scan[fit_start]

### Use the previous fitting x0 as the new start value
# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return idx
# fn_a_pre = f'a_fit_{scan_id}.tiff'
# a_pre = io.imread(in_tiff_dir+fn_a_pre)
# fn_x0_pre = f'x0_fit_{scan_id}.tiff'
# x0_pre = io.imread(in_tiff_dir+fn_x0_pre)
# fn_sigma_pre = f'sigma_fit_{scan_id}.tiff'
# sigma_pre = io.imread(in_tiff_dir+fn_sigma_pre)
# fn_a_dev_pre = f'a_dev_{scan_id}.tiff'
# a_dev_pre = io.imread(in_tiff_dir+fn_a_dev_pre)
# fn_x0_dev_pre = f'x0_dev_{scan_id}.tiff'
# x0_dev_pre = io.imread(in_tiff_dir+fn_x0_dev_pre)
# fn_sigma_dev_pre = f'sigma_dev_{scan_id}.tiff'
# sigma_dev_pre = io.imread(in_tiff_dir+fn_sigma_dev_pre)
# fn_r_square_pre = f'r_square_{scan_id}.tiff'
# r_square_pre = io.imread(in_tiff_dir+fn_r_square_pre)
# idx = find_nearest(eng_scan, x0_pre[x_pixel, y_pixel])
# fit_start = idx-8
# fit_end = idx+10

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))  # The probability density of the normal distribution


       
'''
Gaussian fitting for Field of view
'''
# a_fit = a_pre #np.zeros ((xanes.shape[1], xanes.shape[2]))
# x0_fit = x0_pre #np.zeros ((xanes.shape[1], xanes.shape[2]))
# sigma_fit = sigma_pre #np.zeros ((xanes.shape[1], xanes.shape[2]))
# a_dev = a_dev_pre #np.zeros ((xanes.shape[1], xanes.shape[2]))
# x0_dev = x0_dev_pre #np.zeros ((xanes.shape[1], xanes.shape[2]))
# sigma_dev = sigma_dev_pre #np.zeros ((xanes.shape[1], xanes.shape[2]))
# r_square = r_square_pre #np.zeros ((xanes.shape[1], xanes.shape[2]))

a_fit = np.zeros ((xanes.shape[1], xanes.shape[2]))
x0_fit = np.zeros ((xanes.shape[1], xanes.shape[2]))
sigma_fit = np.zeros ((xanes.shape[1], xanes.shape[2]))
a_dev = np.zeros ((xanes.shape[1], xanes.shape[2]))
x0_dev = np.zeros ((xanes.shape[1], xanes.shape[2]))
sigma_dev = np.zeros ((xanes.shape[1], xanes.shape[2]))
r_square = np.zeros ((xanes.shape[1], xanes.shape[2]))
pixel_time = np.zeros ((xanes.shape[1], xanes.shape[2]))

for i in range(xanes.shape[1]):
    for j in range(xanes.shape[2]):
# for i in range(x_pixel, x_pixel+1):
    # for j in range(y_pixel, y_pixel+1):
        time01 = timeit.default_timer()
        # idx = find_nearest(eng_scan, x0_pre[i, j])
        # fit_start = idx-15   #-15
        # fit_end = idx+10     #+10
        max_idx = np.argmax(xanes[white_0:white_1, i, j]) + white_0
        fit_start = max_idx - start
        fit_end = max_idx + end
        diff = eng_scan[fit_end] - eng_scan[fit_start]
        all_pos = np.all(xanes[fit_start:fit_end,i,j]>0)
        print (f'First fit - Max: {max_idx}; Range: ({fit_start} - {fit_end})')

        
        # if (mask[i,j]==True and mean>6.559 and mean<6.5625 and r_post>1.2 and r_post<2.2 and r_pre>5 and all_pos == True):
        # if (mask[i,j]==True and all_pos == True and xanes[max_idx, i, j]<3 and fit_start>(white_0-start) and fit_end<(white_1+end)):
        if (mask[i,j]==True and all_pos == True and xanes[max_idx, i, j]<3):
            # Fitting range
            x = eng_scan[fit_start:fit_end]
            y = xanes[fit_start:fit_end ,i ,j]
            # weighted arithmetic mean
            mean = sum(x * y) / sum(y)              
            # Fitting function
            sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
            popt,pcov = curve_fit(gaus, x, y, p0=[np.max(y),mean,sigma], maxfev=maxfev)
            # Assign fitting results
            perr = np.sqrt(np.diag(pcov))
            fit = gaus(x, *popt)
            grad = np.gradient(fit)
            positive = len(grad[grad>0])
            negative = len(grad[grad<0])
            first_fit = copy.deepcopy([popt[0], popt[1], popt[2], perr[0], perr[1], perr[2]])

        #### Criteria to auto-tune fitting range ####             
            # fit_start = max_idx - start
            # fit_end = max_idx + end
            num_point = fit_end - fit_start
            c = 0
            
            if (1<(positive-negative)<((start+end)/2) and popt[2]< popt2 and fit[-1]>fit[0]):
                print('Criteria 1 : pass')
            elif (1<(positive-negative)<((start+end)/2) and popt[2]>popt2 and fit[-1]>fit[0]):
                print('Criteria 2')
                while (popt[2]>popt2):              
                    print(f'Checking fitting range: {fit_start+c} - {fit_end+c}')
                    x = eng_scan[(fit_start+c):(fit_end+c)]
                    y = xanes[(fit_start+c):(fit_end+c) ,i ,j]
                    # weighted arithmetic mean
                    mean = sum(x * y) / sum(y)              
                    # Fitting function
                    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
                    popt,pcov = curve_fit(gaus, x, y, p0=[np.max(y),mean,sigma], maxfev=maxfev)
                    # Assign fitting results
                    perr = np.sqrt(np.diag(pcov))
                    fit = gaus(x, *popt)
                    grad = np.gradient(gaus(x, *popt))
                    positive = len(grad[grad>0])
                    negative = len(grad[grad<0])
                    c += 1
                    if ((fit_end-c) < max_idx-10):
                        popt[0], popt[1], popt[2], perr[0], perr[1], perr[2] = first_fit[0], first_fit[1], first_fit[2], first_fit[3], first_fit[4], first_fit[5]
                        break                  
            ### Negative > positive        
            elif (np.negative((start+end)+1)<(positive-negative)<=1):
                print('Criteria 3')
                while (positive<(3+(start+end)/2) or popt[2]>popt2 or fit[-1]<fit[0] or fit[-1]-fit[0]<0.05):              
                    print(f'Checking fitting range: {fit_start-c} - {fit_end-c}')
                    x = eng_scan[(fit_start-c):(fit_end-c)]
                    y = xanes[(fit_start-c):(fit_end-c) ,i ,j]
                    # weighted arithmetic mean
                    mean = sum(x * y) / sum(y)              
                    # Fitting function
                    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
                    popt,pcov = curve_fit(gaus, x, y, p0=[np.max(y),mean,sigma], maxfev=maxfev)
                    # Assign fitting results
                    perr = np.sqrt(np.diag(pcov))
                    fit = gaus(x, *popt)
                    grad = np.gradient(gaus(x, *popt))
                    positive = len(grad[grad>0])
                    negative = len(grad[grad<0])
                    c += 1
                    if ((fit_end-c) < max_idx-10):
                        popt[0], popt[1], popt[2], perr[0], perr[1], perr[2] = first_fit[0], first_fit[1], first_fit[2], first_fit[3], first_fit[4], first_fit[5]
                        break
            
            elif (fit[-1]<fit[0]):
                print('Criteria 4')
                while (fit[-1]<fit[0] or fit[-1]-fit[0]<0.05 or popt[2]>popt2):              
                    print(f'Checking fitting range: {fit_start-c} - {fit_end-c}')
                    x = eng_scan[(fit_start-c):(fit_end-c)]
                    y = xanes[(fit_start-c):(fit_end-c) ,i ,j]
                    # weighted arithmetic mean
                    mean = sum(x * y) / sum(y)              
                    # Fitting function
                    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
                    popt,pcov = curve_fit(gaus, x, y, p0=[np.max(y),mean,sigma], maxfev=maxfev)
                    # Assign fitting results
                    perr = np.sqrt(np.diag(pcov))
                    fit = gaus(x, *popt)
                    grad = np.gradient(gaus(x, *popt))
                    positive = len(grad[grad>0])
                    negative = len(grad[grad<0])
                    c += 1
                    if ((fit_end-c) < max_idx-10):
                        popt[0], popt[1], popt[2], perr[0], perr[1], perr[2] = first_fit[0], first_fit[1], first_fit[2], first_fit[3], first_fit[4], first_fit[5]
                        break     
            ### Positive > negative
            else:
                print('Criteria 5')
                while (negative <= (6) or popt[2]>popt2):              
                    print(f'Checking fitting range: {fit_start+c} - {fit_end+c}')
                    x = eng_scan[(fit_start+c):(fit_end+c)]
                    y = xanes[(fit_start+c):(fit_end+c) ,i ,j]
                    # weighted arithmetic mean
                    mean = sum(x * y) / sum(y)              
                    # Fitting function
                    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
                    popt,pcov = curve_fit(gaus, x, y, p0=[np.max(y),mean,sigma], maxfev=maxfev)
                    # Assign fitting results
                    perr = np.sqrt(np.diag(pcov))
                    grad = np.gradient(gaus(x, *popt))
                    positive = len(grad[grad>0])
                    negative = len(grad[grad<0])
                    c += 1
                    if ((fit_start+c) > max_idx or negative > (positive+negative)*0.8):
                        popt[0], popt[1], popt[2], perr[0], perr[1], perr[2] = first_fit[0], first_fit[1], first_fit[2], first_fit[3], first_fit[4], first_fit[5]
                        break
            
            a_fit[i,j] = popt[0]
            x0_fit[i,j] = popt[1]
            sigma_fit[i,j] = popt[2]
            a_dev[i,j] = perr[0]
            x0_dev[i,j] = perr[1]
            sigma_dev[i,j] = perr[2]
            # Calculating r square
            fitted_result = gaus(x, popt[0], popt[1], popt[2])
            residulas = y - fitted_result
            ss_res = np.sum(residulas**2)
            ss_tot = np.sum((y-np.mean(y))**2)
            r_2 = 1 - (ss_res / ss_tot)
            r2 = f'R\u00b2={r_2:.2f}'
            r_square[i,j] = r_2
        
        else:
            pass  
        time02 = timeit.default_timer()
        fit_time = time02 - time01
        pixel_time[i,j] = fit_time
        print(f'Pixel: ({i}, {j}); fitting time: {fit_time: 2f} seconds.')            

'''
plt.figure()
plt.plot(x,y,'b+:',label='data')
plt.plot(x,fitted_result,'ro:',label='fit\n'+r2)
plt.legend()
plt.title(f'fit peak = {popt[1]: .6f}\u00B1{perr[1]: .6f}', fontsize=12)
fitted_result_all = gaus(x,popt[0], popt[1], popt[2])
plt.figure()
# plt.plot(eng[fit_start-20:fit_end+5], xanes[fit_start-20:fit_end+5,x_pixel,y_pixel], 'b')
plt.plot(eng, xanes[:,x_pixel,y_pixel], 'b')
plt.plot(x,fitted_result_all,'ro:',label='fit')
plt.plot([eng[fit_start], eng[fit_start]], [np.min(xanes[:,x_pixel,y_pixel]), np.max(xanes[:,x_pixel,y_pixel])], 'g--')
plt.plot([eng[fit_end], eng[fit_end]], [np.min(xanes[:,x_pixel,y_pixel]), np.max(xanes[:,x_pixel,y_pixel])], 'g--')
plt.plot(eng[max_idx], xanes[max_idx,x_pixel,y_pixel], 'g*', markersize=16)
plt.show()
'''
 

f1 = f'a_fit_{scan_id}.tiff'
a_fit_32 = a_fit.astype('float32')
out_tiff_1 = out_tiff_dir + f1
io.imsave(out_tiff_1, a_fit_32)

f2 = f'x0_fit_{scan_id}.tiff'
x0_fit_32 = x0_fit.astype('float32')
out_tiff_2 = out_tiff_dir + f2
io.imsave(out_tiff_2, x0_fit_32)

f3 = f'sigma_fit_{scan_id}.tiff'
sigma_fit_32 = sigma_fit.astype('float32')
out_tiff_3 = out_tiff_dir + f3
io.imsave(out_tiff_3, sigma_fit_32)

f4 = f'a_dev_{scan_id}.tiff'
a_dev_32 = a_dev.astype('float32')
out_tiff_4 = out_tiff_dir + f4
io.imsave(out_tiff_4, a_dev_32)

f5 = f'x0_dev_{scan_id}.tiff'
x0_dev_32 = x0_dev.astype('float32')
out_tiff_5 = out_tiff_dir + f5
io.imsave(out_tiff_5, x0_dev_32)    

f6 = f'sigma_dev_{scan_id}.tiff'
sigma_dev_32 = sigma_dev.astype('float32')
out_tiff_6 = out_tiff_dir + f6
io.imsave(out_tiff_6, sigma_dev_32)

f7 = f'r_square_{scan_id}.tiff'
r_square_32 = r_square.astype('float32')
out_tiff_7 = out_tiff_dir + f7
io.imsave(out_tiff_7, r_square_32)

f8 = f'time_{scan_id}.tiff'
time_32 = pixel_time.astype('float32')
out_tiff_8 = out_tiff_dir + f8
io.imsave(out_tiff_8, time_32)

# plt.close('all')
x0_fit[x0_fit == 0] = np.nan
plt.figure()
cmap = plt.cm.coolwarm
plt.imshow(x0_fit, interpolation= None, vmin=8.350, vmax=8.354, cmap = cmap)
# plt.imshow(x0_fit, interpolation= None, cmap = cmap)
# plt.xticks(ticks=[], labels=[])
# plt.yticks(ticks=[], labels=[])
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
plt.show()


