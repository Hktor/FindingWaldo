import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

from scipy.misc import imresize

from skimage import color, io
from skimage.exposure import rescale_intensity
from skimage.transform import rescale
from functions import (detect_edges, move_Window, modify_Intensity, blurry_Img,
                              compare_Template, plot_img_and_tmp, increase_Red_Contrast,
                              window_Color_Analisis, calculate_Best_Match, divide_Img)
                              
from process import (calculate_Rescale, find_Wally_Tmp, where_Is_Waldo)

def find_waldo(img):
    y, x, Ch= img.shape
    R, G, B = 0, 1 ,2
    print(img.shape)

    # Set limits for Color differentiation
    upper, high, lower = 200, 150, 150

    print('Increase Red Contrast...')
    img = increase_Red_Contrast(img, upper, high, lower)
    print('Red Contrast applied...')
    img = imresize(img, 2.0, interp='bilinear', mode=None)
    img = increase_Red_Contrast(img, upper, high, lower)
    ###################   Gray Img   #########################

    print('Increase Red Contrast...')
    img_Gray = color.rgb2gray(img)
    img_Gray = modify_Intensity(img_Gray)
    print('Intensity modified Image...')

    ###################   Wally Temp  ########################
    wally = io.imread('Waldo7.jpg')
    wally_Gray = color.rgb2gray(wally)
    winy, winx = wally_Gray.shape

    ################ Calculate Rescale Factor #################

    scl = calculate_Rescale(img_Gray, wally_Gray, winx, winy)

    print('Rescale Wally :' + str(scl))
    wally = imresize(wally, scl, interp='bilinear', mode=None)
    upp, hgh, low = 230, 200, 170
    wally = increase_Red_Contrast(wally, upp, hgh, low)
    winy, winx, win_ch = wally.shape
    print('Wally rescaled')

    plot_img_and_tmp(img, wally)

    ########    DIVIDE IMAGE TO PERFORM ANALYSIS   ###########

    N = 4 # Number of divisions
    print('Divide image')
    images = divide_Img(img, N)
    print('Image divided succesfully in: ' + str(N))

    results = []
    ########    FIND BEST MATCH BETWEEN IMG & TMP   ##########

    results.append(find_Wally_Tmp(images, wally, winx, winy, win_ch))

    ###################   Wally Temp 2 ########################
    wally = io.imread('Waldo6.jpg')
    wally_Gray = color.rgb2gray(wally)
    winy, winx = wally_Gray.shape

    ################ Calculate Rescale Factor #################
    scl = calculate_Rescale(img_Gray, wally_Gray, winx, winy)

    print('Rescale Wally :' + str(scl))
    wally = imresize(wally, scl, interp='bilinear', mode=None)
    upp, hgh, low = 230, 200, 170
    wally = increase_Red_Contrast(wally, upp, hgh, low)
    winy, winx, win_ch = wally.shape
    print('Wally rescaled')

    ########    FIND BEST MATCH BETWEEN IMG & TMP   ##########

    results.append(find_Wally_Tmp(images, wally, winx, winy, win_ch))
    print('Find Waldo')
    x, y = where_Is_Waldo(img, results, N, winx, winy)
    print(x, y)
    plt.imshow(img)
    plt.show()
    
    return x, y

img = io.imread('27.jpg')
x, y = find_waldo(img)
print(x, y)



