import sys
from PIL import Image

#read sys argument
fileName = sys.argv[1]

#flip image
img = Image.open(fileName).transpose(Image.FLIP_TOP_BOTTOM)

#save image
img.save("ans2.png")