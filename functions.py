# Imports Match Template
import numpy as np
from skimage.feature import match_template
import matplotlib.pyplot as plt
# Imports Detect Circles
from skimage import color
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.transform import hough_circle, hough_circle_peaks
# Imports Detect Lines
from skimage.transform import probabilistic_hough_line, hough_line_peaks
# Imports Filter Picture
from skimage.filters import sobel, prewitt, gaussian, hessian, frangi, gabor
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.util import view_as_windows
# Import for image Resize and Swirl - Compare Template
from scipy.misc import imresize
from skimage.transform import swirl


# Global Variables RGB
R,G,B = 0,1,2

def increase_Red_Contrast(img, up, hg, lw):
    # Increase Red intensity
    RU = np.greater_equal(img[:,:,R],up)
    RH = np.greater_equal(img[:,:,R],hg)
    RL = np.less(img[:,:,R],lw)

    GU = np.greater_equal(img[:,:,G],up)
    GH = np.greater_equal(img[:,:,G],hg)
    GL = np.less(img[:,:,G],lw)

    BU = np.greater_equal(img[:,:,B],up)
    BH = np.greater_equal(img[:,:,B],hg)
    BL = np.less(img[:,:,B],lw)

    RED = RH*GL*BL
    GRN = GH*RL*BL
    BLU = BH*RL*GL
    DRK = RL*GL*BL
    WHT = RU*GU*BU
    YEL = RH*GH*BL
    CYA = RL*GH*BH
    PNK = RH*BH*GL

    img[RED]= 255,0,0
    img[GRN]= 255,255,255 # Set to Background
    img[BLU]= 255,255,255 # Set to Background
    img[DRK]= 0,0,0
    img[CYA]= 255,255,255 # Set to Background
    img[YEL]= 255,255,255 # Set to Background
    img[PNK]= 255,0,255
    img[WHT]= 255,255,255 # Set to Background

    return img

def move_Window(img,shape, step):
    print('Inside Move Window...')
    img_copy = np.array(img, copy=True)
    window = view_as_windows(img_copy, shape, step=int(step))
    return window

def is_Normalized(img):
    if(np.amax(img)<2): return True
    else: return False

def Normalize_RGB(img):
    if(is_Normalized(img)): return img
    else: 
        return modify_Intensity(img)

def is_Red_Sufficient(box, lower_Red_lim, upper_Red_lim, thrsh = 200):
    box_RED = count_RED(box)
    # Verify if patch covers color requirement
    if (box_RED >= lower_Red_lim and 
        box_RED <= upper_Red_lim): return True
    else: return False

def is_Color_Sufficient(box):
    red, drk, wht = count_RED_WHITE_BLACK(box)
    if red < (np.size(box)/3)*0.01: return False # At least there is red
    elif wht > (np.size(box))*0.70: return False # Box is not back ground
    elif drk > (np.size(box))*0.70: return False # Box is not back ground
    else: return True

def count_RED(img, thrsh = 200):
    RH = np.greater_equal(img[:,:,R],thrsh)
    BL = np.less(img[:,:,B],thrsh)
    GL = np.less(img[:,:,G],thrsh)
    RED = RH*BL*GL
    RED = np.array(img[RED])
    return RED.size

def is_Color_Symmetric(img):
    if img.size > 0:
        # The image color must be symmetric
        
        N = 2
        blocks = divide_Img(img,N)
        red = np.zeros(N*N)
        wht = np.zeros(N*N)
        drk = np.zeros(N*N)

        # Difference color - Red, White, Black
        k = 0
        dr = np.zeros(N*N-1)
        dw = np.zeros(N*N-1)
        dd = np.zeros(N*N-1)

        thrsh = np.size(img)*0.10 # Tolerance for difference

        for section in blocks:
            red[k], wht[k], drk[k] = count_RED_WHITE_BLACK(section, 180)

            if k > 0:
                dr[k-1] = np.abs(red[0] - red[k])
                dw[k-1] = np.abs(wht[0] - wht[k])
                dd[k-1] = np.abs(drk[0] - drk[k])

                if (dr[k-1] > thrsh or
                    dw[k-1] > thrsh or 
                    dd[k-1] > thrsh):
                    return False
            k+=1
        return True
    else: return False

def compare_Color(box, tmp_Color, thrsh = 200):
    color_diff = 0
    box_Color = count_RED_WHITE_BLACK(box, thrsh)
    tmp_Color = np.array(tmp_Color)
    box_Color = np.array(box_Color)
    color_diff = (tmp_Color - box_Color)/np.size(box)
    color_diff = np.sum(np.abs(color_diff))
    return color_diff

def count_RED_WHITE_BLACK(img, thrsh = 200):
    # img = Normalize_RGB(img)
    RH = np.greater_equal(img[:,:,R],thrsh)
    BH = np.greater_equal(img[:,:,B],thrsh)
    GH = np.greater_equal(img[:,:,G],thrsh)

    RL = np.less(img[:,:,R],thrsh)
    BL = np.less(img[:,:,B],thrsh)
    GL = np.less(img[:,:,G],thrsh)

    RED = RH*BL*GL
    WHT = RH*BH*GH
    DRK = RL*BL*GL

    RED = np.array(img[RED])
    WHT = np.array(img[WHT])
    DRK = np.array(img[DRK])

    rd = RED.size
    dk = DRK.size
    wh = WHT.size

    return rd, dk, wh


def divide_Img(img, N = 2):
    print('Inside Divide Image...')
    x, y, ch = img.shape
    x_correction = x%N
    y_correction = y%N

    # Define the window size
    wX = int((x-x_correction)/N)
    wY = int((y-y_correction)/N)
    # Verify if correct both sides is even
    if x_correction%2 != 0:
        x_correction+=1
    if y_correction%2 != 0:
        y_correction+=1

    inix = int(x_correction/2)
    endx = inix + wX
    iniy = int(y_correction/2)
    endy = iniy + wY

    div_img = []

    # Split image to Analyse
    for j in range(0,N):
        for i in range(0,N):
            div_img.append(img[inix+wX*(i):endx+wX*(i), iniy+wY*(j):endy+wY*(j), :])
        inix = int(x_correction/2)
        endx = inix +wX
    return div_img

def modify_Intensity(img):
    if img.size > 0:
        max_val = np.amax(img)
        up = max_val * 0.9
        lw = max_val * 0.8
        img[np.greater_equal(img,up)]= max_val
        img[np.less(img,lw)]= 0
    return img

def show_Histograms(images, N, wdth):
    #R, G, B = 0, 1, 2
    k=0
    bins = N*N
    Hist_Red = np.zeros(bins)
    Hist_Blu = np.zeros(bins)
    Hist_Grn = np.zeros(bins)

    for box in images:
        RED = np.greater_equal(box[:,:,R],200)
        BLU = np.greater_equal(box[:,:,B],200)
        GRN = np.greater_equal(box[:,:,G],200)

        RED = np.array(box[RED])
        BLU = np.array(box[BLU])
        GRN = np.array(box[GRN])
        Hist_Red[k]=RED.size
        Hist_Grn[k]=GRN.size
        Hist_Blu[k]=BLU.size
        k+=1
    # Plot Weighted Histogram Red Color
    Sum_All = np.sum(images)
    Hst_Red = Hist_Red/Sum_All
    Hst_Grn = Hist_Grn/Sum_All
    Hst_Blu = Hist_Blu/Sum_All

    aXs = np.linspace(0,bins,bins)
    plt.bar(aXs, Hst_Red, width=wdth, color = 'r')
    aXs = np.linspace(0+.3,bins+0.3,bins)
    plt.bar(aXs, Hst_Blu, width=wdth, color = 'b')
    aXs = np.linspace(0+.6,bins+0.6,bins)
    plt.bar(aXs, Hst_Grn, width=wdth, color = 'g')
    plt.show()


def match_Tmp(img, tmp, offset_X=10, offset_Y = 15):
    print('Inside Match Template...')
    result = match_template(img, tmp)
    h_img, w_img = img.shape
    h_tmp, w_tmp = tmp.shape

    wx = int(w_tmp/2)+1
    wy = int(h_tmp/2)+1

    maxi = np.amax(result)
    xy = np.unravel_index(np.argmax(result), result.shape)
    x, y = xy[::-1]

    if (x < 0): x_lw = 0
    else: x_lw = int(x)
    if (x + w_tmp > w_img): x_hg = int(w_img)
    else: x_hg = int(x + w_tmp)
    if (y < 0): y_lw = 0
    else: y_lw = int(y)
    if (y + h_tmp > h_img): y_hg = int(h_img)
    else: y_hg = int(y + h_tmp)

    mini = np.amin(img)

    img[y_lw:y_hg,x_lw:x_hg] = False # mini

    y = np.array([y_lw, y_hg], dtype=int)
    x = np.array([x_lw, x_hg], dtype=int)

    return img, x, y, maxi

def detect_edges(img):
    max_val = np.amax(img)
    h_thrsh = max_val * 0.80
    l_thrsh = max_val * 0.30
    s = 0.5
    edges = canny(img, sigma=s, low_threshold=l_thrsh, high_threshold=h_thrsh)
    return edges
    
def detect_circles(img, edges, radi, N = 4):
    y, x = img.shape 
    
    # Detect Circles
    hough_radii = np.linspace(radi-1, radi + 3, 5, dtype=int)
    hough_res   = hough_circle(edges, hough_radii)

    # Select the most prominent 5 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                    total_num_peaks=N)

    return cx, cy

def detect_Lines (img, edges, l_length, l_gap, angle):
    lines = probabilistic_hough_line(edges, line_length=l_length,
                                            line_gap=l_gap, theta=angle)
    return lines

def find_Average_Distance(img):
    print('Inside Find Average distance...')
    y, x = img.shape
    edges = detect_edges(img)
    l_length, l_gap = 5, 1
    angle = np.linspace(0,np.pi,10)
    lines = detect_Lines(img, edges, l_length,l_gap, angle)
    center_x, center_y = find_Center_Img(lines)
    qx, qy = find_Quartiles(img)

    distance = []
    ln = 0
    for line in lines:
        p0, p1 = line
        px = np.mean([p0[0],p1[0]])
        py = np.mean([p0[1],p1[1]])

        if (px < center_x + qx and py < center_y + qy and
            px > center_x - qx and py > center_y - qy):
            distance.append(np.abs(px - center_x))
            distance.append(np.abs(py - center_y))
            ln+=1

    return np.mean(distance)

def find_Center_Img(lines):
    x = []
    y = []
    k = 0

    for line in lines:
        p0, p1 = line
        x.append(p0[0])
        x.append(p1[0])
        y.append(p0[1])
        y.append(p1[1])
        k+=1

    center_x = np.mean(x)
    center_y = np.mean(y)
    return center_x, center_y

def find_Quartiles(img):
    y, x = img.shape
    qx = x/4
    qy = y/4
    return qx, qy

def is_Lines_Symmetric(img, img_lines, tmp_lines):
    y, x = img.shape
    cx = x/2
    cy = y/2

    q0, q1, q2, q3 = 0, 0, 0, 0

    for line in img_lines:
        p0, p1 = line
        mx = np.mean([p0[0],p1[0]])
        my = np.mean([p0[1],p1[1]])

        if mx < cx:
            if my < cy: q0+=1
            else: q1+=1
        else:
            if my > cy: q2+=1
            else: q3+=1

    toler = np.size(img_lines)/8

    if (np.abs(q0 - q1) > toler and np.abs(q0 - q2) > toler and np.abs(q0 - q3) > toler or 
        np.abs(q1 - q2) > toler and np.abs(q1 - q3) > toler and np.abs(q1 - q0) > toler or
        np.abs(q2 - q3) > toler and np.abs(q2 - q0) > toler and np.abs(q2 - q1) > toler or
        np.abs(q3 - q0) > toler and np.abs(q3 - q1) > toler and np.abs(q3 - q1) > toler):

        return False
    elif np.size(img_lines) > np.size(tmp_lines)*3: return False
    elif np.size(img_lines) < np.size(tmp_lines)/3: return False
    else: return True

def calculate_Faces_Hat_Stripes_Prob(img, img_lines, center_x, center_y):
    print('Inside Calculate Faces, Hat, Stripes Probabilities...')
    y, x = img.shape # Shape used to scale differences
    k, ln_Y = 0, 0   # k - counter (current line), line to compare
    qx, qy = find_Quartiles(img)
    # Defines Face Lines
    faces = []
    # Defines Stripes Lines
    stripes = []
    lw_strp_thrsh, hg_strp_thrsh = 5, 30 # Distance for detecting stripe lines
    # Defines Hat Lines
    hats = []
    hat_thrsh = 20 # Distance for detecting hat lines

    ln = np.size(img_lines)
    
    for line in img_lines:
        p0, p1 = line
        
        px = np.mean([p0[0], p1[0]])
        py = np.mean([p0[1], p1[1]])
        dy = np.abs(ln_Y - py)
        
        # Face lines located in a radius to close the circles (eyes)
        # And where both circles are located nearly the same 'y' position
        if (px > center_x - qx and px < center_x + qx and
            py > center_y - qy and py < center_y + qy):
            faces.append(np.abs(p0[0]-center_x)/x)
            faces.append(np.abs(p1[0]-center_x)/x)
            faces.append(np.abs(px - center_x)/x)
            faces.append(np.abs(p0[1]-center_y)/y)
            faces.append(np.abs(p1[1]-center_y)/y)
            faces.append(np.abs(py - center_y)/y)

        # Horizontal lines, below eyes, which are separated 
        # an accetable distance = probable stripes
        if (p0[1] - p1[1] == 0 and py > center_y + qy and
            dy < hg_strp_thrsh and dy > lw_strp_thrsh):
            stripes.append(np.abs(p0[0]-center_x)/x)
            stripes.append(np.abs(p1[0]-center_x)/x)
            stripes.append(np.abs(px - center_x)/x)
            stripes.append(np.abs(p0[1]-center_y)/y)
            stripes.append(np.abs(p1[1]-center_y)/y)
            stripes.append(np.abs(py - center_y)/y)

        # Horizontal Hats, lines above img center, which are separated 
        # an accetable distance = hat probability
        if (py < center_y - qy):
            hats.append(np.abs(p0[0]-center_x)/x)
            hats.append(np.abs(p1[0]-center_x)/x)
            hats.append(np.abs(px - center_x)/x)
            hats.append(np.abs(p0[1]-center_y)/y)
            hats.append(np.abs(p1[1]-center_y)/y)
            hats.append(np.abs(py - center_y)/y)
        ln_Y = py
    if not faces:
        faces = -1
    else:
        faces = np.mean(faces)
    if not hats:
        hats = -1
    else:
        hats = np.mean(hats)
    if not stripes:
        stripes = -1
    else:
        stripes = np.mean(stripes)

    return faces, stripes, hats, ln

def compute_Diff(img_var, tmp_var):
    if img_var == -1 or tmp_var == -1:
        return -1
    else:
        return np.abs(img_var - tmp_var)

def blurry_Img(img, sig):
    filtered_img = gaussian(img, sigma=sig, multichannel=True)
    return filtered_img

def plot_img_and_tmp(img, tmp):
    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(1, 2, 1, adjustable='box-forced')
    ax2 = plt.subplot(1, 2, 2, adjustable='box-forced')

    ax1.imshow(tmp, cmap='gray')
    ax1.set_title('Template')

    ax2.imshow(img, cmap='gray')
    ax2.set_title('Image')
    plt.show()

def window_Color_Analisis(patches, img_Gray, wx, wy, lower_Red_Limit, upper_Red_Limit, step):
    print('Inside Window Analisis...')
    color_dismatch = [] # List containing possibles color matches
    count_clm = 0 # Count for window position
    count_row = 0 # Count for window position
    count_red = 0
    # Loop windows to verify which ones fullfil the color requirement:
    for box_row in patches:
        count_clm = 0
        for box in box_row:
            if is_Color_Sufficient(box)==True:
                count_red+= 1
            else:
                color_dismatch.append([count_row, count_clm])
            count_clm+=1
        count_row+=1

    for y_dismatch, x_dismatch in color_dismatch:
        x0 = int(x_dismatch * step)
        x1 = int(x0 + wx)

        y0 = int(y_dismatch * step)
        y1 = int(y0 + wy)

        img_Gray[y0:y1,x0:x1] = False

    return img_Gray

def compare_Template(img_Gray, tmp, scl):
    print('Inside compare template...')
    posible_match = []
    count = 0
    for i in range(0,scl.size):
        print('Rescale Wally :' + str(scl[i]))

        tmp_Gray = np.array(tmp, copy=True)
        tmp_Gray = imresize(tmp_Gray, scl[i], interp='bilinear', mode=None)

        img, x, y, maxi  = match_Tmp(img_Gray, tmp_Gray)
        posible_match.append([x, y, maxi])

        img, x, y, maxi  = match_Tmp(img_Gray, tmp_Gray)
        posible_match.append([x, y, maxi]) 
        
        tmp_swirl = swirl(np.array(tmp_Gray, copy=True), cval=0, # cval = 1 if white
                                    strength=1, radius=150, rotation=np.pi/6)

        img, x, y, maxi  = match_Tmp(img_Gray, tmp_swirl)
        posible_match.append([x, y, maxi])

        tmp_swirl = swirl(np.array(tmp_Gray, copy=True), cval=0, # cval = 1 if white
                                    strength=1, radius=130, rotation=np.pi/4)
        
        img, x, y, maxi  = match_Tmp(img_Gray, tmp_swirl)
        posible_match.append([x, y, maxi])

    return posible_match

def calculate_Best_Match(img, tmp, posible_match, scl, thrsh):
    print('Inside Calculate Best match...')
    img_y, img_x, img_ch = img.shape
    k, count = 0, 1

    colors = []
    shapes = []
    stripes = []
    faces = []
    patches = []
    probs = []
    cdnts = []

    results = []

    angle = np.linspace(0,np.pi, 10)
    lin_thrsh, lin_length, lin_gap = 0.5, 3, 1
    circles_number = 2

    # Loop windows to verify which ones fullfil the color requirement:
    for i, j, shp in posible_match:

        tmp_Scaled = np.array(tmp, copy=True)
        tmp_Scaled = imresize(tmp_Scaled, scl[k], interp='bilinear', mode=None)

        box = img[np.amin(j):np.amax(j),np.amin(i):np.amax(i),:]
        box = modify_Intensity(box)

        upper_Red_lim = tmp_Scaled.size*0.60
        lower_Red_lim = tmp_Scaled.size*0.02

        if is_Color_Symmetric(box):
            print('Symmetric Color...')
            tmp_Color = count_RED_WHITE_BLACK(tmp_Scaled, thrsh)

            # Compute Statistics Lines Patch - Template

            tmp_Gray = color.rgb2gray(tmp_Scaled)
            tmp_edges = detect_edges(tmp_Gray)
            tmp_lines = detect_Lines(tmp_Gray, tmp_edges, lin_length, lin_gap, angle)

            box_Gray = color.rgb2gray(box)
            box_Gray = modify_Intensity(box_Gray)
            box_edges = detect_edges(box_Gray)
            box_lines = detect_Lines(box_Gray, box_edges, lin_length, lin_gap, angle)

            if is_Lines_Symmetric(box_Gray, box_lines, tmp_lines):
                print('Symmetric Line...')

                tmp_cx, tmp_cy = find_Center_Img(tmp_lines)
                tmp_fa, tmp_ha, tmp_st, tmp_ln = calculate_Faces_Hat_Stripes_Prob(tmp_Gray, tmp_lines, tmp_cx, tmp_cy)

                box_cx, box_cy = find_Center_Img(box_lines)
                box_fa, box_ha, box_st, box_ln = calculate_Faces_Hat_Stripes_Prob(box_Gray, box_lines, box_cx, box_cy)

                # Compute Differences Patch and Template
                col = compare_Color(box, tmp_Color, thrsh)
                fac = compute_Diff(box_fa,tmp_fa)
                stp = compute_Diff(box_st,tmp_st)
                hat = compute_Diff(box_ha,tmp_ha)
                lin = compute_Diff(box_ln,tmp_ln)

                # Calculate Weighted Difference
                w_prob = calculate_Weighted_Match(shp, col, fac, stp, hat, lin)

                coord = calculate_Ini_coordinates(j, i) # y, x

                patches.append(box)
                probs.append(w_prob)
                shapes.append(shp)
                faces.append(fac)
                cdnts.append(coord)
                colors.append(col)
                stripes.append(stp)
            else:
                print('Lines are not symmetric...')
        else:
            print('Color is not symmetric')
        
        if(count%4==0):
            k+=1
        count+=1

    # Find Best Matches for possible Waldos
    
    if not patches:
        print('No matches found')
        boxis = -1
        probis = -1
        shapis = -1
        colori = -1
        total_prob = -1
        coordi = -1
        location = -1

    else:
        matches, location = find_Best_Match(patches, probs, colors, shapes, stripes, faces, cdnts)

        # Generate new images to iterate and verify Waldo Identity
        boxis, probis, shapis, colori, coordi, total_prob = generate_Match_Patches(patches, matches, shapes, 
                                                                                   probs, colors, cdnts)
    
    return boxis, probis, shapis, colori, coordi, location, total_prob

def calculate_Weighted_Match(shapes, colors, faces, stripes, hat, lin):
    print('Inside Calculate Weighted Match...')
    wshp = 20 # Weigth shape, because correlation values vary from 1, -1
    wcol = 20 # Weigth color, because of normalized value 0, 1
    wlin = 0.2 # Weigth lines
    wfac = 0.2 # Weight face

    if stripes != -1: wstp = 0.1 # Weight stripes
    else: wstp = 0.0

    if hat != -1: what = 0.1 # Weight hat
    else: what = 0.0

    w_prob = ((1 - shapes)*wshp + colors*wcol + faces*wfac + hat*what + stripes*wstp + lin*wlin)
    return w_prob

def find_Best_Match(patch, probs, colors, shapes, strips, faces, coordinates):
    print('Inside Find Best Match...')
    best_probs = np.argsort(probs)
    best_color = np.argsort(colors)
    best_shape = np.argsort(shapes)
    best_shape = best_shape[::-1]
    
    # print('Color match: '   + str(best_color))
    # print(colors) 
    # print('Shape match: '   + str(best_shape))
    # print(shapes)
    # print('Probs match: '   + str(best_probs))
    # print(probs)

    best_match = []
    
    # Look for intersection after taking the best (0), thus 1
    for i in range (0,best_color.size):
        best_c = best_color[:i]
        best_p = best_probs[:i]
        best_s = best_shape[:i]
        
        best = np.intersect1d(best_c, best_p)
        best = np.intersect1d(best, best_s)
        if (best.size >= 1):
            break

    for b in best:
        best_match.append(b)
    
    location = coordinates[best_shape[0]] # y, x
    best_match = np.unique(best_match)
    print(best_match)
    return best_match, location

def generate_Match_Patches(patches, matches, shapes, probs, colors, cdnts):
    print('Inside Generate Match Patches')
    boxis = []
    probis = []
    shapis = []
    colori = []
    coordi = []

    k = 0
    for b in matches:
        boxis.append(patches[b])
        probis.append(probs[b])
        shapis.append(shapes[b])
        colori.append(colors[b])
        coordi.append(cdnts[b])
        k+=1

    if k == 1 : missing_matches = 0
    elif k == 2 : missing_matches = -1
    else: missing_matches = -2

    if not probis: acc_prob = 1000
    else: acc_prob = np.amin(probis)

    return boxis, probis, shapis, colori, coordi, acc_prob

def calculate_Ini_coordinates(j, i):    
    if np.size(j) > 0 or  np.size(i) > 0: 
        x = np.amin(i)
        y = np.amin(j)
        return np.array([y, x])     
    else:
        return -1 
        
def generate_Accumulated(var, missing_matches):
    if not var: total = -1
    else: total = np.amin(var)
    return total
    
