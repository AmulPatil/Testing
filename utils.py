import numpy as np
from PIL import Image, ImageOps
from fastbook import *
from google.colab import files
import glob, os
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image


def stain_recognition():
  folders_1 = glob.glob('*.jpg')
  folders_2 = glob.glob('*.jpeg')
  single_list = [*folders_1,*folders_2]
  for image_path in single_list[:1]:
      im = Image.open(image_path)
      im.convert('RGB').save("1.jpg","JPEG") #this converts png image as jpeg
  img = cv2.imread('1.jpg', cv2.IMREAD_UNCHANGED)
  cv2_imshow(img)
  pred,pred_idx,probs = learn.predict(img)
  #remove jpg file for next file
  folders_1 = glob.glob('*.jpg')
  folders_2 = glob.glob('*.jpeg')
  filelist = [*folders_1,*folders_2]
  for f in filelist:
      os.remove(f)
  print('------------------------------------')
  print(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
  print('------------------------------------')
  return 