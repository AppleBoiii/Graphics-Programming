import struct
import numpy as np
import cv2

def show(img, waitTime=0):
    cv2.imshow('image', img)
    cv2.waitKey(waitTime)
    if not waitTime:
        cv2.destroyAllWindows()

def writeBTCH(img, filename="test"):
    if len(img.shape) > 2:
        raise Exception("Error: Image is not black and white.")
    data = encode(img)
    f = open(filename+".BTCH", "wb")
    f.write(b"BTCH")

    # print(data)
    # print(len(data))
    h, w = img.shape
    f.write(struct.pack("<H", w))
    f.write(struct.pack("<H", h))

    for thing in data:
        f.write(struct.pack("B", thing))
    f.close()
      

def readBTCH(filename="test"):
    f = open(filename+".BTCH", "rb")
    x = f.read(4)
    if x!= b"BTCH":
        raise Exception("Error: File is not a BTCH file.")
    
    w, h = struct.unpack("<2H", f.read(4))
    print(w, h)
    encoded_data = f.read()
    img = []
    x = 0

    for data in encoded_data:
        img += [x]*data
        x = 255-x

    img = np.array(img, dtype=np.uint8)
    img = np.reshape(img, (h, w))
    return img

def encode(img):
    h, w = img.shape
    diff = np.diff(img.ravel())
    # x = 0
    loc = np.where(diff!=0)[0]
    loc = np.pad(loc, (1,1), 'constant', constant_values=(-1, h*w-1))
    runs = np.diff(loc) #length of the runs
    print(runs)
    out = []

    for run in runs:
        while run >255:
            out += [255, 0]
            run -= 255
        out.append(run)
    
    print(out[:4])
    return out

img = cv2.imread("test.bmp", 0)
writeBTCH(img, "epic")
img = readBTCH("epic")
show(img)

print('done!')
