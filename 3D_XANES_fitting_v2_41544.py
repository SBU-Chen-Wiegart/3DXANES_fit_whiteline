# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:50:52 2019

@author: chenlin
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
import timeit

# %matplotlib qt
# %matplotlib notebook
# %matplotlib inline
#plt.close('all')

start_time = timeit.default_timer()

MnK=6.539
Mn_eng_list_7 = list(np.arange(MnK-0.082, 6.535,0.01)) + list(np.arange(6.535+0.0005, 6.547, 0.0005)) + list(np.arange(6.547+0.0005, 6.563, 0.00025)) + list(np.arange(6.563+0.00025, MnK+0.08, 0.01)) + list(np.arange(MnK+0.09, MnK +0.288, 0.025))
#del Mn_eng_list_7[8]
#Mn_eng_list_4 = list(np.arange(MnK-0.082, 6.535,0.01)) + list(np.arange(6.535+0.0005, 6.57, 0.0005)) + list(np.arange(6.572, MnK+0.08, 0.007)) + list(np.arange(MnK+0.09, MnK +0.288, 0.025))

#inputdir = 'D:\\20190816_FXI\\xanes_3D\\xanes_assemble\\'
inputdir = 'D:\\20191205_FXI\\41544_41651\\crop_align_bin4_2Dxanes\\'
out_tiff_dir = 'D:\\20191205_FXI\\41544_41651\\eng_tiff\\'
out_png_dir = 'D:\\20191205_FXI\\41544_41651\\eng_png\\'
eng_shift = 2.14/1000    # Date: 2019/08/16:2.91 eV, 2019/12/05: 2.14 eV, 2020/02/10: 1.38 eV


'''
1. Check the starting energy of the 1st tomopraphy
2. Select fitting range
'''
#fn = 'ali_recon_24850_bin_1.h5'
#h5_file = inputdir + fn
#h5 = h5py.File(h5_file, 'r')
#print(list(h5.keys()))
#eng = np.array(h5['X_eng'])
#print(eng)

eng_scan = np.array(Mn_eng_list_7[:]) + eng_shift
eng_scan = np.around(eng_scan, 6)
start_eng = 6.55664 #6.550  #6.555
end_eng = 6.56464 #6.5725  #6.563
index_start = np.where(eng_scan == start_eng)[0]
fit_start = index_start[0]
index_end = np.where(eng_scan == end_eng)[0]
fit_end = index_end[0]


'''
Read assembled 2D XANES from tiff
'''
all_slices = range(0, 476)
#all_slices = range(250, 251)
#slice_num = 463
#condition = 'C95_LMO_C2_cut2'   #Sample condition


for slice_num in all_slices:
    
    plt.close('all')
    #tiff_file = 'xanes_2D_slice_' + f'{slice_num:04d}' + '.tiff'
    tiff_file = 'xanes_2D_slice_' + "%03d" %slice_num + '.tiff'
    tiff = io.imread(inputdir + tiff_file)
    # The data have beeen took -log during reconstruction.
    log_tiff_0 = tiff
    
    
    
    '''    
    Generate mask by setting a threshold value with averaging insterest post-edge
    '''    
    # Post_edge : > 6.562 
    #index_post = np.where(eng_scan > end_eng)
    index_post = np.where(eng_scan > 6.562)
    post_mean = np.mean(log_tiff_0[index_post], axis = 0)
    #fn = inputdir + 'post_mean_' + f'{slice_num:04d}' + '.tiff'
    #io.imsave(fn, post_mean)
    #plt.figure()
    #plt.imshow(post_mean, interpolation= None)
    #plt.hist(post_mean, normed=True, bins=20)
    #plt.ylim(0, 1500)
    
    threshold = 0.001
    mask = post_mean >= threshold
    #mask = mask.astype(int)
    #plt.figure()
    #plt.imshow(mask, interpolation= None)
    #fn = inputdir + 'mask_' + f'{slice_num:04d}' + '.tiff'
    #io.imsave(fn, mask)
    
    
    # '''
    # Fit and plot spectrum from a single pixel
    # '''
    # x_pixel = 96
    # y_pixel = 193
    # plt.figure()
    # plt.plot(eng_scan, log_tiff_0[:,x_pixel,y_pixel], 'b')
    # #imag_name = 'Single_XANES_' + condition + '_(' + str(x_pixel) + ', ' + str(y_pixel) + ')' 
    # #plt.savefig(imag_name, dpi = 300)

    # # max_idx = np.argmax(log_tiff_0[:, x_pixel, y_pixel])
    # # fit_start = max_idx-25
    # # fit_end = max_idx+14
    # x = eng_scan[fit_start:fit_end]
    # y = log_tiff_0[fit_start:fit_end ,x_pixel ,y_pixel]
    
    # # weighted arithmetic mean
    # mean = sum(x * y) / sum(y)
    # sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
    
    
    # def gaus(x,a,x0,sigma):
    #     return a*exp(-(x-x0)**2/(2*sigma**2))  # The probability density of the normal distribution
    
    
    # popt,pcov = curve_fit(gaus,x,y,p0=[np.max(y),mean,sigma], maxfev=1000000)
    # fitted_result = gaus(x,*popt)
    # perr = np.sqrt(np.diag(pcov))
    # residulas = y - fitted_result
    # ss_res = np.sum(residulas**2)
    # ss_tot = np.sum((y-np.mean(y))**2)
    # r_2 = 1 - (ss_res / ss_tot)
    # r2 = f'R\u00b2={r_2:.2f}'
    # # r_square[i,j] = r_2
    
    # plt.figure()
    # plt.plot(x,y,'b+:',label='data')
    # plt.plot(x,fitted_result,'ro:',label='fit\n'+r2)
    # plt.legend()
    # plt.title(f'fit peak = {popt[1]: .6f}\u00B1{perr[1]: .6f}', fontsize=12)
    # #plt.title('Fig. 3 - Fit for Time Constant')
    # #plt.xlabel('Time (s)')
    # #plt.ylabel('Voltage (V)')
    # plt.show()
    # #imag_name = 'Gau_fitting_' + condition + '_(' + str(x_pixel) + ', ' + str(y_pixel) + ')'  
    # #plt.savefig(imag_name, dpi = 300)
    
    
    
       
    '''
    Gaussian fitting for Field of view
    '''
   
    def gaus(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))  # The probability density of the normal distribution
    
    
    a_fit = np.zeros ((log_tiff_0.shape[1], log_tiff_0.shape[2]))
    x0_fit = np.zeros ((log_tiff_0.shape[1], log_tiff_0.shape[2]))
    sigma_fit = np.zeros ((log_tiff_0.shape[1], log_tiff_0.shape[2]))
    a_dev = np.zeros ((log_tiff_0.shape[1], log_tiff_0.shape[2]))
    x0_dev = np.zeros ((log_tiff_0.shape[1], log_tiff_0.shape[2]))
    sigma_dev = np.zeros ((log_tiff_0.shape[1], log_tiff_0.shape[2]))
    r_square = np.zeros ((log_tiff_0.shape[1], log_tiff_0.shape[2]))
    pixel_time = np.zeros ((log_tiff_0.shape[1], log_tiff_0.shape[2]))
    
    
    for i in range(log_tiff_0.shape[1]):
        for j in range(log_tiff_0.shape[2]):
            time01 = timeit.default_timer()
            max_idx = np.argmax(log_tiff_0[:, i, j])
            # fit_start = max_idx-15
            # fit_end = max_idx+16           
            if (mask[i,j]==True and eng_scan[max_idx]>6.559 and eng_scan[max_idx]<6.563):
                # Fitting range
                x = eng_scan[fit_start:fit_end]
                y = log_tiff_0[fit_start:fit_end ,i ,j]
                # weighted arithmetic mean
                mean = sum(x * y) / sum(y)
                sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
                # Fitting function
                maxfev=1000000
                popt,pcov = curve_fit(gaus, x, y, p0=[np.max(y),mean,sigma], maxfev=maxfev)
                # Assign fitting results
                perr = np.sqrt(np.diag(pcov))
                a_fit[i,j] = popt[0]
                x0_fit[i,j] = popt[1]
                sigma_fit[i,j] = popt[2]
                a_dev[i,j] = perr[0]
                x0_dev[i,j] = perr[1]
                sigma_dev[i,j] = perr[2]
                # Calculating r square
                fitted_result = gaus(x, *popt)
                residulas = y - fitted_result
                ss_res = np.sum(residulas**2)
                ss_tot = np.sum((y-np.mean(y))**2)
                r_2 = 1 - (ss_res / ss_tot)
                #r2 = f'R\u00b2={r_squared:.2f}'
                r_square[i,j] = r_2
             
        
            else:
                pass  
            time02 = timeit.default_timer()
            fit_time = time02 - time01
            pixel_time[i,j] = fit_time            
                
                

    f1 = 'a_fit_' + "%03d" %slice_num + '.tiff'
    a_fit_32 = a_fit.astype('float32')
    out_tiff_1 = out_tiff_dir + f1
    io.imsave(out_tiff_1, a_fit_32)
    
    f2 = 'x0_fit_' + "%03d" %slice_num + '.tiff'
    x0_fit_32 = x0_fit.astype('float32')
    out_tiff_2 = out_tiff_dir + f2
    io.imsave(out_tiff_2, x0_fit_32)
    
    f3 = 'sigma_fit_' + "%03d" %slice_num + '.tiff'
    sigma_fit_32 = sigma_fit.astype('float32')
    out_tiff_3 = out_tiff_dir + f3
    io.imsave(out_tiff_3, sigma_fit_32)
    
    
    f4 = 'a_dev_' + "%03d" %slice_num + '.tiff'
    a_dev_32 = a_dev.astype('float32')
    out_tiff_4 = out_tiff_dir + f4
    io.imsave(out_tiff_4, a_dev_32)
    
    f5 = 'x0_dev_' + "%03d" %slice_num + '.tiff'
    x0_dev_32 = x0_dev.astype('float32')
    out_tiff_5 = out_tiff_dir + f5
    io.imsave(out_tiff_5, x0_dev_32)    
    
    f6 = 'sigma_dev_' + "%03d" %slice_num + '.tiff'
    sigma_dev_32 = sigma_dev.astype('float32')
    out_tiff_6 = out_tiff_dir + f6
    io.imsave(out_tiff_6, sigma_dev_32)
    
    f7 = 'r_square_' + "%03d" %slice_num + '.tiff'
    r_square_32 = r_square.astype('float32')
    out_tiff_7 = out_tiff_dir + f7
    io.imsave(out_tiff_7, r_square_32)
    
    f8 = 'time_' + "%03d" %slice_num + '.tiff'
    time_32 = pixel_time.astype('float32')
    out_tiff_8 = out_tiff_dir + f8
    io.imsave(out_tiff_8, time_32)
    
    plt.close('all')
    x0_fit[x0_fit == 0] = np.nan
    plt.figure()
    cmap = plt.cm.coolwarm
    plt.imshow(x0_fit, interpolation= None, vmin=6.560, vmax=6.562, cmap = cmap)
    plt.xticks(ticks=[], labels=[])
    plt.yticks(ticks=[], labels=[])
    ax = plt.gca()
    ax.set_facecolor('k')
    cbar = plt.colorbar(ticks=[6.560, 6.5605, 6.561, 6.5615, 6.562])
    cbar.set_label('White Line (keV)', labelpad = 10, fontsize=15, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    #cbar.ax.set_yticklabels([6.558, 6.559, 6.56])
    #imag_name = inputdir + 'Gaussain_' + slice_num
    imag_name = 'Gaussain_whiteline_' + "%03d" %slice_num
    #out_png_dir = '/home/xf18id/Desktop/users/2019Q3/KAREN_Proposal_305052/41544_41651/fitted_png/'
    out_png = out_png_dir + imag_name
    plt.savefig(out_png, dpi = 300)
    ps = 'Fitting of slice ' + "%03d" %slice_num + ' is done!'
    print(ps)
    
    stop_time = timeit.default_timer()
    elapse_time = stop_time - start_time
    print(f'Elapsed time for fitting slice {slice_num}: {elapse_time: .2f} seconds.') 




