import numpy as np
import cv2

# while True:
#     # img = img[:-1]/2+img[1:]/2 #look at difference between / and //
#     img = img[:, :-1]/2+img[:, 1:]/2 #look at difference between / and //
#     show(np.uint8(img), 100)

def show(img, waitTime=0):
    cv2.imshow('output.png', img)
    cv2.waitKey(waitTime)
    if not waitTime:
        cv2.destroyAllWindows()


def filter(img, k): #doesn't deal with edges, things can shrink.
    h, w = k.shape
    H, W = img.shape[:2]
    out = np.zeros((H-h+1, W-w+1, 3), dtype=np.float64)

    for j in range(h):
        for i in range(w):
            kv = k[j][i]
            patch = img[j:H-h+1+j, i:W-w+1+i] #think of 3x3 or 9x9 image. goes and gets every combination of a -2 window on those 
            out += kv*patch

    return np.uint8(out)

def gaussianKernel(n):
    if n <= 1:
        return np.ones((1, 1), dtype=np.float64)

    kernel = np.zeros((n, 1), dtype=np.float64)
    babyKernel = gaussianKernel(n-1)
    kernel[1:] += babyKernel
    kernel[:-1] += babyKernel
    kernel /= kernel.sum()

    return kernel


img = cv2.imread('image1.png')

# # kernel = np.float64([[1, 1, 1,], [1, 1, 1], [1, 1, 1]])/9 #this is a basic averaging blur filter
# kernel = np.ones((101, 1), dtype=np.float64) #big box blurs like this suck, use gaussian filters
# kernel = cv2.getGaussianKernel(ksize=301, sigma=0)
# # print(kernel)
# # kernel = kernel@kernel.T
# # print(kernel*256)

# out = cv2.filter2D(img, -1, kernel, borderType = cv2.BORDER_CONSTANT)
# # out = cv2.sepFilter2D(img, -1, kernel, kernel.T)
# # show(np.uint8(out))


# while True:
#     # out = filter(out, kernel)
#     out = cv2.filter2D(out, -1, kernel, borderType=cv2.BORDER_CONSTANT) #-1 = same depth, number of bits
#     show(np.uint8(out!=last)*255.0, 1)
#     last = out*1
#     # show(out, 15)


#i think this is edge finding. look at wikapedia (sobel filter)
#This does a cool kernel / line finding filter thing on the image
# kernel = np.float64([[0, 0, 0,], [1, 0, -1], [0, 0, 0 ]])
# out = cv2.filter2D(out, None, kernel)
# show(out)
# out = cv2.fitler2D(out, None, kernel.T)
# show(out)

#for sharpening, look at more kernels. think of subtracting a blurred image from an original image. 

#adding the kernel values up to 1 is for blurring, adding all the values up to 0 is more for finding edges and lines

k = gaussianKernel(3)
print(k)

'''
1x5 @ 5x1 = 1x1
5x1 @ 1x5 = 5x5
print(k@k.T)

basic edge filter
0 0 0
1 0 -1
0 0 0

can combine with gaussian and get (does two filters at once) (similar to sobel filter)
1 0 -1
2 0 -2
1 0 -1

'''