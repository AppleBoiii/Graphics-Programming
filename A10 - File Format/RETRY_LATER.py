from math import frexp
import struct
import cv2
import numpy as np

class Node:
    def __init__(self, freq, symbol, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
        self.huff = '' #tree direction
    # def __str__(self):
    #     x = ""
    #     if self.left != None:
    #         x += f"[{self.left}]" #this calls the __str__ func for self.left
        
    #     x += str(self.symbol)

    #     if self.right != None:
    #         x += f"({self.right})"

    #     return x
def treeToString(root: Node, string: list):
    # base case
    if root is None:
        return
 
    # push the root data as character
    string.append(str(root.symbol))
 
    # if leaf node, then return
    if not root.left and not root.right:
        return
 
    # for left subtree
    string.append('(')
    treeToString(root.left, string)
    string.append(')')
 
    # only if right child is present to
    # avoid extra parenthesis
    if root.right:
        string.append('(')
        treeToString(root.right, string)
        string.append(')')
    
    return ''.join(string)

def show(img, waitTime=0):
    cv2.imshow('image', img)
    cv2.waitKey(waitTime)
    if not waitTime:
        cv2.destroyAllWindows()

def writeSWRD(img, filename):
    if len(img.shape)>2:
        raise Exception("Image is not 1 channel")

    f = open(filename+".SWRD", "wb") #writes binary
    f.write(b"SWRD") #writes a byte-string
    h,w = img.shape
    f.write(struct.pack("<I", w))
    f.write(struct.pack("<I", h))

    img = img.ravel() #2D -> 1D
    #print(img.tobytes())
    f.write(img.tobytes)
    f.close()

def writeBTCH(img, filename="test"):
    if len(img.shape)>2:
        raise Exception("Error: Image type contains more than one channel.")
    
    f = open(filename+".BTCH", "wb")
    f.write(b"BTCH")
    h, w = img.shape
    f.write(struct.pack("<H", w))
    f.write(struct.pack("<H", h))

    img = img.ravel()
    for i, pixel in enumerate(img):
        if pixel == 0:
            for j in range(8, -1, -1):
                try:
                    subArr = img[i:i+j]
                    if np.count_nonzero(subArr) == len(subArr):
                        x = treeMap[f'{str(len(subArr))}b']
                        # print(x)
                        f.write(x)
                except:
                    continue
        else:
            for j in range(8, -1, -1):
                try:
                    subArr = img[i:i+j]
                    if np.count_nonzero(subArr) == len(subArr):
                        x = treeMap[f'{str(len(subArr))}w']
                        # print(x)
                        f.write(x)
                except:
                    continue

def readBTCH(filename, tree):
    f = open(filename+".BTCH", "rb")
    magic_nums = f.read(4)
    if magic_nums != b"BTCH":
        raise Exception("Error: Cannot read non-BTCH file")
    w, h = struct.unpack("<2H", f.read(4))
    print(w, h)
    data = f.read(w*h)
    img = []
    current = tree
    for i in range(h*w):
        if data[i] == 0:
            current = current.left
        else:
            current = current.right
        
        if not current.left and not current.right:
            symbol = current.symbol
            if symbol[1] == 'b':
                img.append([0]*int(symbol[0]))
            else:
                img.append([255]*int(symbol[0]))
            current = tree
    
    return np.reshape((np.uint8(img)), (h, w))

def readSWRD(filename):
    f = open(filename+".SWRD", "rb")
    x = f.read(4)
    if x!= b"SWRD":
        raise Exception("Not a SWRD file.")
    w, h = struct.unpack("<2I", f.read(8))
    data = f.read(w*h)
    data = np.frombuffer(data, dtype=np.uint8)

    return np.reshape(data, (h, w))

def formatTree(node, val=''):
    newVal = val + str(node.huff)
  
    #edge node case
    if(node.left):
        formatTree(node.left, newVal)
    if(node.right):
        formatTree(node.right, newVal)
    
    #no edge node case
    if(not node.left and not node.right):
        treeMap[f"{node.symbol}"] = newVal.encode()
        # print(f"{node.symbol} -> {newVal}")

def makeHuffManTree(counter):
    symbols = list(counter.keys())[2:]
    frequencies = list(counter.values())[2:]
    tree = []

    for symbol, frequency in zip(symbols, frequencies):
        tree.append(Node(frequency, symbol))
    
    while len(tree) > 1:
        tree = sorted(tree, key=lambda x:x.freq) #sorts nodes in ascending order, based on key=frequency
        left, right = tree[0], tree[1] #smallest two nodes
        left.huff, right.huff = 0, 1

        newNode = Node(left.freq+right.freq, left.symbol+right.symbol, left, right) #sum of the two smallest nodes
        tree.remove(left)
        tree.remove(right)
        tree.append(newNode)
    
    return tree[0]


    '''
get frequency of byte sequences
so, how many times there are of 1 byte
2 byte, 3 byte, or...8 byte in a row
    '''
def getFrequency(img, x=255):
    counter = {
        "0w": 0,
        "0b": 0,
        "1b": 0,
        "2b": 0,
        "3b": 0,
        "4b": 0,
        "5b": 0,
        "6b": 0,
        "7b": 0,
        "8b": 0,
        "1w" : 0,
        "2w": 0,
        "3w": 0,
        "4w": 0,
        "5w": 0,
        "6w": 0,
        "7w": 0,
        "8w": 0
    }

    img = img.ravel()
    img = img.tostring()
    x = (img*1).split(b"\x00")

    for i, c in enumerate(x):
        if len(c)>=9:
            x.append(c[8:])
            x[i] = c[:8]
    for c in x:
        counter[str(len(c))+'w'] += 1 
    
    y = (img*1).split(b'\xff')
    for i, c in enumerate(x):
        if len(c)>=9:
            x.append(c[8:])
            x[i] = c[:8]
    for c in x:
        counter[str(len(c))+'b'] += 1 
    
    return counter


img = cv2.imread("xkcd.bmp", 0)
frequencies = getFrequency(img)

tree = makeHuffManTree(frequencies)
treeMap = {
    "1b": 0,
    "2b": 0,
    "3b": 0,
    "4b": 0,
    "5b": 0,
    "6b": 0,
    "7b": 0,
    "8b": 0,
    "1w": 0,
    "2w": 0,
    "3w": 0,
    "4w": 0,
    "5w": 0,
    "6w": 0,
    "7w": 0,
    "8w": 0
}
formatTree(tree) #updates treeMap 
print(treeMap)
# writeBTCH(img)
# d_img = readBTCH("test", tree)


print("Done")

# string = []
# print(treeToString(tree, string))
# print(c)



# s = struct.pack("<H", 504)  #packs 504 into little-endian 2 byte unsigned short

#seward way (run-length encoding)
# h, w = img.shape
# diff = np.diff(img.ravel())
# loc = np.where(diff!=0)[0]
# loc = np.pad(loc, (1, 1), 'constant', constant_values=(-1, h*w-1))
# runs = np.diff(loc) #length of runs
# out = []
# for run in runs:
#     while run>255:
#         out += [255, 0]
#         run -= 255
#     out += [run] #(same as .append(run))'
# data = []
# x = 255
# for byte in out:
#     data += [x]*byte
#     x = 255-x

# img = np.uint8(data).reshape(h, w)
# #then show(img)
# print(len(out)) #rough file size


'''
4BYTES MAGIC NUMBERS: SWRD
4BYTES WIDTH, LITTLE ENDIAN uINT:
4BYTES HEIGHT, LITTLE ENDING uINT:
1BYTE for 1 PIXEL(L-R, TOP-BOTTOM)


take an xkcd (saved as monochrome bitmap), aliased
compress into file format either the same size or smaller

'''

