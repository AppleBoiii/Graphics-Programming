import numpy as np
from PIL import Image

def main():
	im = Image.open('image.png')
	
	for i in range(360):
		im = im.rotate(i)
		im.show()
		i += 30
if __name__ == '__main__':
	main()
