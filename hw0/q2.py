import sys
from PIL import Image

#read sys argument
fileName = sys.argv[1]

#flip image
img = Image.open(fileName).transpose(Image.ROTATE_180)

#save image
img.save("ans2.png")