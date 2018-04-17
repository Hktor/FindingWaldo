import numpy as np
import numpy.ma as ma

from skimage import color
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

from functions import (detect_edges, move_Window, modify_Intensity,
                              compare_Template, plot_img_and_tmp, find_Average_Distance,
                              window_Color_Analisis, calculate_Best_Match)

R, G, B = 0, 1 ,2

def calculate_Rescale(img, tmp, winx, winy):
    smp = img[50:50+winy, 50:50+winx]

    print('Calculate Sample Typical Distance...')
    smp_typ_dist = find_Average_Distance(smp)
    tmp_typ_dist = find_Average_Distance(tmp)

    print('Typical Distance image: ' + str(smp_typ_dist))
    print('Typical Distance wally: ' + str(tmp_typ_dist))

    if tmp_typ_dist > smp_typ_dist: scl = smp_typ_dist/tmp_typ_dist
    else: scl = tmp_typ_dist/smp_typ_dist

    return scl


def find_Wally_Tmp(images, tmp, tmp_x, tmp_y, tmp_ch):
    ############### REFINE TMP EDGES ####################

    tmp_Gray = color.rgb2gray(tmp)
    tmp_Gray = modify_Intensity(tmp_Gray)
    tmp_Gray = detect_edges(tmp_Gray)

    k=0

    boxes = []
    probs = []
    shaps = []
    colrs = []
    coord = []
    section = []

    total_probability = []
    position = []

    results = []

    for image in images:
            ############### GRAY IMAGE #######################
            print('Increase Red Contrast...')
            image_Gray = color.rgb2gray(image)
            image_Gray = modify_Intensity(image_Gray)
            print('Intensity modified Image...')

            ############### REFINE EDGES ####################
            print('Refine Edges Image...')
            image_Gray = detect_edges(image_Gray)

            # ############## DIVIDE IMAGE IN WINDOWS TO COMPARE RED COLOR ################
            step, winx, winy = tmp_x/2, tmp_x, tmp_y

            p_shape = np.array((winy, winx, tmp_ch))
            print('Divide image for color analysis...')
            patches = move_Window(image, p_shape, step)
            patches = np.squeeze(patches, axis=(2))
            print('Image has been divided into: ' + str(patches.shape))

            ###### CHECK IF RED COLOR IS SUFFICIENT IN PATCH OTHERWAY REMOVE IT ########

            lower_Red_Limit = tmp.size*0.15 # Considering min red = 15% template
            upper_Red_Limit = tmp.size*0.60 # Considering max red = 60% template

            print('Analyze patches color match...')
            image_Gray = window_Color_Analisis(patches, image_Gray, winx, winy,
                                               lower_Red_Limit, upper_Red_Limit, step)
            print('Analysis finished')

            # plt.imshow(image_Gray)
            # plt.show()
            ############# COMPARE TEMPLATE tmp FOR POSSIBLE MATCHES #############
            scl = np.linspace(0.4, 0.9, 10)
            print('Match Template...')
            posible_match = compare_Template(image_Gray, tmp_Gray, scl)
            print('Match Template finished')

            ############### COMPARE COLOR POSSIBLE MATCHES #######################
            thrsh = 200   # Threshold for color intensity RGB
            print('Calculate Possible Match difference...')
            # calculate_Best_Match(image, tmp, posible_match, scl, thrsh, radi)
            boxis, probis, shapis, colori, coordi, location, total_prob = calculate_Best_Match(image, tmp, posible_match, 
                                                                                               scl, thrsh)
            if boxis == -1:
                print('No matches in this section')
                total_probability.append(1000)
                position.append([-1,-1])
            else:
                print('Matches in this section')
                for box in boxis:
                    boxes.append(box)
                for prob in probis:
                    probs.append(prob)
                for shap in shapis:
                    shaps.append(shap)
                for col in colori:
                    colrs.append(col)
                for crd in coordi:
                    coord.append(crd)

                total_probability.append(total_prob)
                position.append(location)

            print('Possible Match difference calculated')
            print('Position: ' + str(location))
            print('Probable: ' + str(total_probability))

    for box in boxes:
        plt.imshow(box)
        plt.show()


    return total_probability, position

def where_Is_Waldo(img, results, N, winx, winy): 
    x0, y0, ch = img.shape
    print(img.shape)
    print('Inside Where is Waldo')
    NN = N*N
    
    final_probabt = np.zeros((2, NN))
    final_psition = np.zeros((2, NN, 2))
    k=0
    for result in results:
        total_prob, location = result
        for i in range(0, NN):
            final_psition[k, i] = location[i]
            final_probabt[k, i] = total_prob[i]
        k+=1
    
    coordinates = []
    probability = []

    for i in range(0,NN):
        x1, y1, p = select_Coordinates_Prob(final_psition[0,i], final_psition[1,i], 
                                             final_probabt[0, i], final_probabt[1,i], winx, winy)
        coordinates.append([x1, y1])
        probability.append(p)

    indx_prob_sorted = np.argsort(probability, 0)
    print(probability)
    print(indx_prob_sorted)

    indx = indx_prob_sorted[0]
    row = indx%N
    clm = indx//N
    print(row, clm)
    
    if row > 0: y_init = y0*(row/N)
    else: y_init = 0
    if clm > 0: x_init = x0*(clm/N)
    else: x_init = 0

    print(indx, x_init, y_init)
    print(coordinates[indx])

    x, y = coordinates[indx]

    x+= x_init
    y+= y_init

    return x, y

def select_Coordinates_Prob(coord1, coord2, prob1, prob2, winx, winy):
    y0, x0 = coord1
    y1, x1 = coord2

    if y0 < 0 or x0 < 0:
        print('Empty coordinates')
        x, y, p = -1, -1, -1
    else:
        if y1 < 0 or x1 < 0:
            print('Empty coordinates')
            print('Empty coordinates')
            x, y, p = -1, -1, -1
        else:
            if prob1 > prob2:
                x = int(x1)
                y = int(y1)
                p = int(prob2)
            else:
                x = int(x0)
                y = int(y0)
                p = prob1

    return x, y, p