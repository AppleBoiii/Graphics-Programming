import numpy as np
import cv2
import scipy.signal as signal
import scipy.ndimage.morphology as morphology

def show(img, waitTime=0):
    if img.min() < 0 or img.max() > 255 or img.max() - img.min() < 20:
        img=normalize(img)
    cv2.imshow('image', img)
    cv2.waitKey(waitTime)
    if not waitTime:
        cv2.destroyAllWindows()

def saveImg(img, name="output.png"):
    cv2.imwrite(name, img)
    print("Saved!")

def normalize(img):
    img = img*1.0
    img=img-np.min(img)
    # print("NORMALIZING!!!")
    img/=img.max()
    img*=255.99999
    return np.uint8(img)

def detectCirlces(input, d=0, downSize=4, N=.7, cannyThreshold1=50, cannyThreshold2=20):
    img = input[::downSize, ::downSize]*1
    h, w = img.shape[:2]
    if d==0:
        d = min(h, w)
    
    # print(d)
    edgeMap = cv2.Canny(img, cannyThreshold1, cannyThreshold2)
    
    Y, X, D = np.mgrid[:d, :d, :d]
    distances = np.hypot(Y-d//2, X-d//2) #Y-d//2, X-d//2 is the center of the circle
    circles = ((D>3)*(distances>D/2-1)*(distances<D/2+1))/(D+.1) #so the circle is wherever distance > r-1 and < r+1
    circles = circles[:, :, ::-1] #flips le cone


    votes = signal.correlate(edgeMap[:,:,None]*1.0, circles*1.0) #similar to cv2 filter2D or cv2.matchTemplate
    show(votes[:, :, 20])
    saveImg(votes[:, :, 20], "Output\\circle_votes.png")
    _, _, v_d = votes.shape

    dilated_votes = morphology.grey_dilation(votes, (5, 5, 5))

    peaks = ((dilated_votes==votes)*(votes>votes.max()*N))

    circled_img = img*1
    peaks_y, peaks_x, peaks_d = np.where(peaks)
    for i, j, r in zip(peaks_x, peaks_y, peaks_d//2):
        cv2.circle(circled_img, ((i-v_d//2),(j-v_d//2)), r, (0,255,0), 2)

    show(circled_img)
    saveImg(circled_img, "Output\\circled_img.png")

def detectShape(input, patch, downSize=4, N=.7):
    img = input[::downSize, ::downSize]*1
    patch = cv2.blur(patch, (5, 5))
    h, w = img.shape[:2]
    edgeMap = cv2.Canny(img, 50, 20)

    votes = signal.correlate(edgeMap*1.0, cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)*1.0)
    show(votes)
    saveImg(votes, "Output\\shape_votes.png")
    dilated_votes = cv2.dilate(votes,(5, 5, 5))
    peaks = ((dilated_votes==votes)*(votes>votes.max()*N))

    drawn_img = img*1
    peaks_y, peaks_x = np.where(peaks)
    ph, pw = patch.shape[:2]
    alpha = (1-patch)
    for i, j in zip(peaks_x, peaks_y):
        if j-ph >= 0 and j < h and i-pw >= 0 and i < w:
            background = alpha*(drawn_img[j-ph:j, i-pw:i])
            # show(background)
            drawn_img[j-ph:j, i-pw:i] = patch + background


    show(drawn_img)
    saveImg(drawn_img, "Output\\drawn_img.png")


custom_background = cv2.imread("Input\\background.png")
circle_background = cv2.imread("Input\\coins.png")
custom_patch = cv2.imread("Input\\patch.png")

detectCirlces(circle_background, downSize=2, N=.65)  
detectShape(custom_background, custom_patch, downSize=1, N=.70 )
