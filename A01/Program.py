import numpy as np
import cv2


'''
my utility funcs
'''
def makeSlider(start=50, range=1000):

    cv2.namedWindow('slider-window')
    cv2.createTrackbar('contrast-factor', 'slider-window', start, range, slider_show)
    cv2.waitKey(0)

def slider_show(n):
    cv2.imshow('slider window', contrast(img, n/10))
    print(n)

def show(img, waitTime=0):
    cv2.imshow('output.png', img)
    cv2.waitKey(waitTime)
    if not waitTime:
        cv2.destroyAllWindows

def saveImg(img, name="output.png"):
    cv2.imwrite(name, img)
    print("Saved!")

def normalize(img):
    img = img - np.min(img) #makes sure the lowest value is 0 in the image
    img = img / img.max() #now you get values between 0 and 1
    img *= 255.9999 

    return img
'''
assignment funcs
'''
def greyscale(img, weight=0, save=False): #greyscale, color is a % between black and white
    out = img*1
    if len(out.shape) < 3 or out.shape[2] < 3: #in case the image is already greyscaled
        return out
    #steps: weight the color channel values and divide to get an image with one channel
    #one way: #70%G, 20%R, 10%B

    '''
    i dont think these have any noticeable differences but i got the second one from researching it so i thought
    might as well
    '''
    if not weight:
        out = img[:, :, 0]*0.1 + img[:, :, 1]*0.7 + img[:, :, 2]*0.2
    else:
        out = (img[:, :, 0]*0.0722 + img[:, :, 1]*0.7152 + img[:, :, 2]*0.2126)
    
    if save:
        saveImg(out, "output\\greyscale.png")

    return np.uint8(out)

def blackWhite(img, threshold, save=False): #ONLY black and white
    #the cutoff is at what color value the pixel becomes white or black
    cutoff = threshold + 128
    out = greyscale(img)
    # print(f"{out.max()}")
    out[out > cutoff] = 255
    out[out <= cutoff] = 0

    if save:
        saveImg(out, f"output\\blackWhite{cutoff}.png")
    
    return out

def desaturate(img, percent=0.5, save=False):
    if len(img.shape) < 3:
        out = (greyscale(img)*percent)+img*(1-percent)
        return np.uint8(out)

    out = (greyscale(img)[:, :, None]*percent)+img*(1-percent)
    #add a certain % of grey to the color image to make it desaturate
    #multiply greyscale by factor to get a % of grey and add that
    #to the other percentage of the color to get 1 image

    if save:
        saveImg(out, f"output\\desaturated-{percent}.png")
    
    return np.uint8(out)

def contrast(img, factor=1, save=False):
    out = img*1.0
    out = (out - 128)*factor + 128
    
    out[out > 255] = 255    #prevent overflow
    out[out < 0] = 0

    if save:
        saveImg(out, f"output\\contrast{factor}.png")

    return np.uint8(out)

def tint(img, color=128, percent=0.5, save=False):
    #tint is actually like desaturate, it's an image of the color added to the other image by a certain percent
    color_img = img*1
    color_img[:, :] = color
    out = np.uint8((color_img*percent)+img*(1-percent))
    
    if save:
        saveImg(out, f"output\\tint{color},{percent}.png")

    return out



img = cv2.imread("input\\image1.png")
# contrast(img, 0.5, True)
# show(img)
# makeSlider()
 