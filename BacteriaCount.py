import cv2
from PIL import Image
import numpy as np
from scipy import ndimage
from skimage import measure, color, io
import torch, celldetection as cd
from matplotlib import pyplot as plt

# Load Cell Detection Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = cd.fetch_model('ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c', check_hash=True).to(device)
model.eval()

def cropFrame(img):
    height, width, channels = img.shape
    croppedFrame = img[0:img.shape[0], 0:int(img.shape[1]/2)]
    return croppedFrame

framenum = 1
gif = Image.open("/Users/agastyabhardwaj/Downloads/bacteria-animation.gif")
gif.save('test1.png')
for frames in range(1, gif.n_frames):
    gif.seek(gif.tell()+1)
    framenum += 1
    gif.save('test' + str(framenum) + '.png')

bacteriaCount = []
for frame in range(1, 41):
    img = cv2.imread('test'+ str(frame) +'.png')
    img = cropFrame(img)
    
    with torch.no_grad():
        x = cd.to_tensor(img, transpose=True, device=device, dtype=torch.float32)
        x = x / 255 
        x = x[None]
        y = model(x)

    contours = y['contours']
    num_contours = len(contours[0])
    bacteriaCount.append(num_contours)

    #Code To Display Contours Found

    #for n in range(len(x)):
        #cd.imshow_row(x[n], x[n], figsize=(16, 9), titles=('input', 'contours'))
        #cd.plot_contours(contours[n])
        #plt.show()
with open("count.txt", "w") as file:
  for item in bacteriaCount:
    file.write(str(item) + "\n")