import numpy as np
import cv2

def show(img, waitTime=0):
    cv2.imshow('output.png', img)
    cv2.waitKey(waitTime)
    if not waitTime:
        cv2.destroyAllWindows()

def saveImg(img, name="output.png"):
    cv2.imwrite(name, img)
    print("Saved!")

def getT():
    return np.float64([[1, 0, 0], \
                      [0, 1, 0], \
                      [0, 0, 1]])

def mirror(img, axis=0, save=False):
    #0=x, 1=y
    # show(img)
    h, w = img.shape[:2]

    mirror = np.float64([[-1, 0, w], \
                        [0, 1, 0], \
                        [0, 0, 1]])
    
    # mirror = np.float64([[1, 0, 0], \
    #                     [0, -1, h], \
    #                     [0, 0, 1]])

    out = cv2.warpPerspective(img, mirror, (w, h))
    if save:
        saveImg(out, "Output\\mirror.png")
    # show(out)
    return out

def rotate(img, degrees=30, save=False): 
    h, w = img.shape[:2]
    angle = (degrees*np.pi)/180 #in radians
    cos = np.cos(angle)
    sin = np.sin(angle)

    r, t1, t2 = getT(), getT(), getT()
    t1[0, 2], t1[1, 2] = -w, -h
    t2[0, 2], t2[1, 2] = w, h

    ''' 
    t1 = [1, 0, -w]
         [0, 1, -h]
         [0, 0, 1]

    r = [cos, -sin, 0]
        [sin, cos , 0]
        [0,    0  , 1]

    t2 = [1, 0, w]
         [0, 1, h]
         [0, 0, 1]
    '''

    r[0, 0], r[0, 1] = cos, -sin
    r[1, 0], r[1, 1] = sin, cos

    out = cv2.warpPerspective(img, t2@r@t1, (w, h))
    if save:
        saveImg(out, "Output\\rotate1.png")
    show(out)
    return out

def rotate2(img, degrees=30, save=False):
    '''
    do the same thing as above, but fix the (wxh) and t matrix to make it fit within window margins
    '''
    h, w = img.shape[:2]
    angle = (degrees*np.pi)/180 #in radians
    cos = np.cos(angle)
    sin = np.sin(angle)

    r, t1, t2, t3= getT(), getT(), getT(), getT()
    t1[0, 2], t1[1, 2] = -w, -h
    t2[0, 2], t2[1, 2] = w, h

    r[0, 0], r[0, 1] = cos, -sin
    r[1, 0], r[1, 1] = sin, cos

    a = w*sin
    b = w*cos

    dX = -(w-b)
    t3[0, 2] = dX
    dW = int(h*sin+w*cos)

    dY = int((a+(h*cos))-h)
    t3[1, 2] = dY #e = dH and dY
    dH = dY+h

    '''
    t3 = [1, 0, dX]
         [0, 1, dY]
         [0, 0,  1]
    '''

    m = t3@t2@r@t1

    out = cv2.warpPerspective(img, m, (dW, dH))
    if save:
        saveImg(out, f"Output\\rotate2{angle}.png")
    show(out)
    return out

def getSquare(img, size, start=0):
    slice = img[start:start+size, start:start+size]
    return slice

def extractSquareQuestion(img1, img2, img3, save=False):
    ih, iw = img3.shape[:2]

    sq = getSquare(img, 200, 200)
    h1, w1 = sq.shape[:2]
    sq2 = getSquare(img2, 200, 200)
    h2, w2 = sq2.shape[:2]

    #coords of cube faces
    #f1 = 167, 236: 391, 204 - 410, 433 : 176, 493
    #f2 = 23, 170: 154, 246 - 158, 489: 27, 400
    coords_1 = [[167, 236], [391, 204], [410, 433], [176, 493]]
    coords_2 = [[23, 170], [154, 246], [158, 489], [27, 400]]

    #raccoons
    srcPoint = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
    destPoint = np.float32(coords_1)
    M = cv2.getPerspectiveTransform(srcPoint, destPoint)
    out1 = cv2.warpPerspective(sq, M, (iw, ih), flags=cv2.INTER_NEAREST)

    #gremlin
    srcPoint2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]])
    destPoint2 = np.float32(coords_2)
    N = cv2.getPerspectiveTransform(srcPoint2, destPoint2)
    out2 = cv2.warpPerspective(sq2, N, (iw, ih), flags=cv2.INTER_NEAREST)

    mask = (out1 > 0)
    mask2 = (out2 > 0)

    p1 = (out1*mask+img3*(1-mask))
    
    out = out2*mask2+p1*(1-mask2)
    if save:
        saveImg(out, "Output\\extractSquare.png")
    show(np.uint8(out))
    return np.uint8(out)

def flatten(img, save=False):
    #corners = [(135, 65), (424, 81), (414, 572), (148, 504)]
    #destPoint = corners+34
    h, w = img.shape[:2]
    w_to_h = 8 /(12+(1/8)) #constants from common cereal box size

    srcPoint = np.float32([(135, 65), (424, 81), (414, 572), (148, 504)])
    destPoint =  np.float32([(0, 0), (int(h*w_to_h), 0), (int(h*w_to_h), h), (0, h)])
    M = cv2.getPerspectiveTransform(srcPoint, destPoint)

    out = cv2.warpPerspective(img, M, (int(h*w_to_h), h))

    if save:
        saveImg(np.uint8(out), "Output\\flatten.png")

    show(np.uint8(out))
    return np.uint8(out)

def bonus(img, img2, save=False):
    h0, w0 = img.shape[:2]

    flat = flatten(img)
    h, w = flat.shape[:2]
    sq = getSquare(img2, 200, 200)

    flat[h//2 - 100: h//2 + 100, w//2 - 100: w//2 + 100] = sq
    show(flat)
    saveImg(flat, "Output\\flatwithgremelin.png")

    srcPoint = np.float32([(0, 0), (w, 0), (w, h), (0, h)])
    destPoint = np.float32([(135, 65), (424, 81), (414, 572), (148, 504)])

    M = cv2.getPerspectiveTransform(srcPoint, destPoint)
    out = cv2.warpPerspective(flat, M, (w0, h0))

    mask = cv2.warpPerspective(flat*0+255, M, (w0, h0), flags=cv2.INTER_NEAREST)
    mask = mask > 0
    out = out*mask+img*(1-mask)

    if save:
        saveImg(np.uint8(out), "Output\\deflatten.png")

    show(np.uint8(out))

    
img = cv2.imread("Input\\image1.png")
img2 = cv2.imread("Input\\image2.png")
img3 = cv2.imread("Input\\image3.png")
img4 = cv2.imread("Input\\image4.png")


mirror(img, 1)
# rotate(img, save=True)
# rotate2(img, 30, save=True)
# rotate2(img, 69, save=True)
# extractSquareQuestion(img, img2, img3, save=True) #can be improved
# flatten(img4, save=True)
# bonus(img4, img2)






#get perspective transform
# srcPoint = np.float32() #original coords
# dstPoint = np.float32() #where you want them to be after
# M = cv2.getPerspectiveTransform()
# out = cv2.warpPerspective, img1, M, None, flags=cv2.INTER_Nearest)
#can't just add images
#mask = out > 0 (this is a bad choice for a mask when by itself)
#instead make a version of the mask (the transformed image) that is the image but only white -> mask = cv2.warpPerspective(img*0+255, M, None, flags=cv2.INTER_NEAREST)
# mask = mask > 0
# show(np.uint8(out*mask+img1*(1-mask))) #has a few problems with it

#another way to fix the anti-aliasing and make the composite better
#whiteOutVersion = img1*0+255
#whiteOutVersion[:10] = 0
#whiteOutVersion[-10:0] = 0
#whiteOutVersion[:, :10] = 0
#whiteOutVersion[:, -10:0] = 0
#mask = warPerspective(whiteOutVersion)
#show(np.uint8(out*mask/255.0+img1*(255-mask)/255.0)) #has a few problems with it