import cv2
import numpy as np

#70%G, 20%R, 10%B

'''
don't copy sewárds funcs
'''

class Image:
    def __init__(self, x):
        if type(x) == str:
            self.img = cv2.imread(x)
        else:
            self.img = x
    
    def __add__(self,x): #assume x is scalar
        out = self.img*1 + x
        out[out<x] = 255
        return Image(out)
    
    def __getitem__(self, m): #allows Image[y, x] to happen instead of Image[y][x]
        y, x = m
        return self.img[y,x]

    def show(self, waitTime = 0):
        if self.img.min()<0 or self.img.max()>255:
             self.img = normalize(self.img)
        cv2.imshow('output.png', self.img)
        cv2.waitKey(waitTime)
        if not waitTime:
            cv2.destroyAllWindows()
    
    def normalize(self):
        img = self.img*1
        img = img - np.min(img) #makes sure the lowest value is 0 in the image
        img = img / img.max() #now you get values between 0 and 1
        img *= 255.9999 

        return img
    
    def greyscale(self, modifySelf = False):
        img = self.img*1
        return Image(np.uint8(img[:, :, 0]*0.1+img[:, :, 1]*0.7+img[:, :, 2]*0.2/3))
    
def show(img, waitTime=0):
    if img.min()<0 or img.max()>255:
        img = normalize(img)
    cv2.imshow('output.png', img)
    cv2.waitKey(waitTime)
    if not waitTime:
        cv2.destroyAllWindows()

def normalize(img):
    img = img - np.min(img) #makes sure the lowest value is 0 in the image
    img = img / img.max() #now you get values between 0 and 1
    img *= 255.9999 

    return img

def greyFilter(img):
    return np.uint8(img[:, :, 0]*0.1+img[:, :, 1]*0.7+img[:, :, 2]*0.2/3) #this gives a pretty good output, still wrong way needs to be weighted (good weights at top, research more)

def blackWhiteFilter(img, threshold=128):
    output = greyFilter(img)
    # output = img[:, :, 1]*1 # green makes up 70% of the data so this is a very haphazard way
    output[output > threshold] = 255
    output[output <= threshold] = 0

    '''
    this bad
    output[output > 128] = 255
    output[output <= 128] = 0
    '''
    return output

def contrast(img, factor = 2): 
    out = (img - 128.0)*factor + 128
    out[out < 0] = 0
    out[out > 255] = 255
    out = out.astype(np.uint8())
    # print(out)
    
    return out


# img = cv2.imread('ronnie1.jpg')
# img = blackWhiteFilter(img)
# grey = greyFilter(img)
# halfandhalf = grey[:, :, None]//2 + img//2 #np.new_axis also works, but None just makes a new dimension of none values
# contrasted = contrast(img, 100)
# show(contrasted)
# show(img)
# show(halfandhalf)

def test(v):
    b = 100
    # print(v)
    cv2.imshow('slider window', contrast(img, v/b)) #make v negative for cool

def makeSlider():
    cv2.namedWindow('slider window')
    cv2.createTrackbar('contrast_factor', 'slider window', 0, 255, test)
    cv2.waitKey()

img = cv2.imread("\\Users\\staln\\OneDrive\\Desktop\\Computer Science\\Graphics Programming\\A01\\input\\image1.png")
# makeSlider()
img = greyFilter(img)
show(img)

'''
contrast:
min = 118
max = 138
double contrast = 108 - 148

118 - 128 = -10*2 = -20 -> (128 + -20 = 108 )
138 - 128 = 10*2 = 20 -> (128 + 20 = 148)
'''