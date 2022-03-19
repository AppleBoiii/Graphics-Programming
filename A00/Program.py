import numpy as np
import cv2
from numpy.lib.function_base import extract

img1 = cv2.imread('Input\\image1.png')
img2 = cv2.imread('Input\\image2.png')
flag = cv2.imread('\\Users\\staln\\OneDrive\\Desktop\\Computer Science\\Graphics Programming\\A00\\Input\\flag.png')
test = cv2.imread('Input\\test.png')

H, W = flag.shape[:2]
#singapore flag
SINGAPORE_RED = [54, 36, 238]
GREEN = [0, 255, 0]
#red color = 54, 36, 238

def show(img):
    
    cv2.imshow('output.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows

def normalize(img):
    img = img - np.min(img) #makes sure the lowest value is 0 in the image
    img = img / img.max() #now you get values between 0 and 1
    img *= 255.9999 

    return img

def switch_RG_channels(img): #pic_1_a = switch_RG_channels(img1)
    g = img[:, :, 1]
    r = img[:, :, 2]

    new_img = img*1
    new_img[:, :, 2], new_img[:, :, 1] = g, r

    cv2.imwrite('Output\\pic_1_a.png', new_img) 

    #return new_img

def extract_blue(img): #pic_1_b = extract_blue(img2)
    new_img = img[:, :, 0]*1

    cv2.imwrite('Output\\pic_1_b.png', new_img) 

    # return new_img

def invert_green(img): #pic_1_c = invert_green(img1)
    new_img = img*1
    new_img[:, :, 1] = 255 - new_img[:, :, 1]

    cv2.imwrite('Output\\pic_1_c.png', new_img) 
    # return new_img

def add100(img): #pic_1_d = addd100(img1)
    new_img = img*1
    new_img[new_img >= 155] = 255
    new_img[new_img < 255] += 100

    # print(np.max(img[:, :, 1]))
    # print(np.min(new_img[:, :, 1]))

    cv2.imwrite('Output\\pic_1_d.png', new_img)

    return new_img

def centered100(img): #pic_2_a = centered100(img1)
    new_img = img*1
    h, w = img.shape[:2]
    new_img[(h//2)-50:(h//2)+50, (w//2)-50:(w//2)+50, 1] = 255

    cv2.imwrite('Output\\pic_2_a.png', new_img) 
    # return new_img
    
def replaceCenter(img1, img2): #pic_2_b = replaceCenter(img1, img2)
    new_img = img2*1
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    new_img[(h2//2)-50:(h2//2)+50, (w2//2)-50:(w2//2)+50] = img1[(h1//2)-50:(h1//2)+50, (w1//2)-50:(w1//2)+50]
    cv2.imwrite('Output\\pic_2_b.png', new_img)
    # show(new_img)

def stats(img1): 
    numOfPixels = 471*373
    stat_file = open('Output\\stats.txt', 'w')
    stat_file.write(f"Number of Pixels: {numOfPixels}\n")
    minIntensity = np.amin(img1)
    maxIntensity = np.amax(img1)
    stat_file.write(f"min: {minIntensity}, max:{maxIntensity}\n")
    stn_deviation = np.std(img1)
    mean_intensity = np.mean(img1)
    stat_file.write(f"Standard Deviation: {stn_deviation}\n")
    stat_file.write(f"Mean Intensity: {mean_intensity}\n")
    stat_file.write("The standard deviation is 62. the mean intensity is 120 ~ 121, meaning that the average color value of all the pixels is 121. So, 68 percent of the pixels will have values 62 +- of 121. ")
    #the standard deviation is 62. the mean intensity is 120 ~ 121, meaning that the average color value of all the pixels is 121. So, 
    # 68% of the pixels will have values 62 +- of 121. 
    stat_file.close()

def star(cx, cy, img, angle = 0):
    y, x = np.mgrid[:H, :W]
    starX = cx
    starY = cy
    starRadius = 39
    theta = np.arctan2(y-starY,x-starX) + (np.pi/2) + angle
    theta %= np.pi*2/5
    r = np.hypot(y-starY,x-starX)
    theta = np.minimum(theta,np.pi*2/5-theta)
    m = np.tan(18*np.pi/180)
    xp = r*np.cos(theta)
    yp = r*np.sin(theta)
    img[yp<-m*(xp-starRadius)] = 255

    return img
    
def flagify(): #pic_4_a = flagify()
    y, x = np.mgrid[:H, :W]
    new_flag = np.zeros((H, W, 3), dtype = np.uint8)
    new_flag[:H//2, :], new_flag[H//2:, :] = SINGAPORE_RED, [255, 255, 255]
    c1 = (x - 249)**2 + (y - 218)**2 < 152**2 
    c2 = (x - 327)**2 + (y - 218)**2 < 152**2 
    new_flag[c1], new_flag[c2] = 255, SINGAPORE_RED

    new_flag = star(345, 125, new_flag)
    new_flag = star(265, 189, new_flag)
    new_flag = star(421, 189, new_flag)
    new_flag = star(295, 290, new_flag)
    new_flag = star(406, 290, new_flag)
    # show(new_flag)
    cv2.imwrite('Output\\pic_4_a.png', new_flag) 

    return new_flag

#fix this
def diff(img1, img2): #pic_4_b = diff(img1, img2)
    orig_diff = (img1*1.0 - img2)
    new_diff = normalize(orig_diff)
    new_diff = new_diff.astype(np.uint8)

    cv2.imwrite('Output\\pic_4_b.png', new_diff)





'''
Just uncomment the thing you need to do. 
For the diff, the flagify() line has to
also be uncommented.
'''
# switch_RG_channels(img1)
# extract_blue(img2)
# invert_green(img1)
# add100(img2)
centered100(img1)
replaceCenter(img1, img2)
# stats(img1)

# new_flag = flagify()
# diff(flag, new_flag)


