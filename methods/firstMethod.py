#Author: Alican Akca

import os
import cv2
import numpy as np
from PIL import Image

os.chdir("/")

def toThePixL(square):
  files = os.listdir(f"{os.getcwd()}/original")
  for index, file in enumerate(files):
    img = Image.open(f"{os.getcwd()}/original/{file}")
    img = img.convert("RGB")
    img = img.resize((64, 64), Image.LANCZOS)
    ImgW, ImgH = img.size
    epicAlgorithm(square,ImgW,ImgH,file,img)

def numOfSquaresFunc(square: int,ImgW,ImgH):
  numOfSquaresW = round((ImgW / square) - 1)
  numOfSquaresH = round((ImgH / square) - 1)
  return (numOfSquaresW,numOfSquaresH) , square, square//2

def epicAlgorithm(square: int,ImgW,ImgH,file,img):
  pixValues = []

  for j in range(1,numOfSquaresFunc(square,ImgW,ImgH)[0][1]+1):

    for i in range(1,numOfSquaresFunc(square,ImgW,ImgH)[0][0]+1):
      
      pixValues.append((img.getpixel((
            i * numOfSquaresFunc(square,ImgW,ImgH)[1] - numOfSquaresFunc(square,ImgW,ImgH)[2],
            j * numOfSquaresFunc(square,ImgW,ImgH)[1]- numOfSquaresFunc(square,ImgW,ImgH)[2])),
            (i * numOfSquaresFunc(square,ImgW,ImgH)[1] - numOfSquaresFunc(square,ImgW,ImgH)[2],
            j * numOfSquaresFunc(square,ImgW,ImgH)[1] - numOfSquaresFunc(square,ImgW,ImgH)[2])))
 
  background = 255 * np.ones(shape=[ImgW - numOfSquaresFunc(square,ImgW,ImgH)[1], ImgH - numOfSquaresFunc(square,ImgW,ImgH)[1], 3], dtype=np.uint8)         
  
  for pen in range(len(pixValues)):
    
    cv2.rectangle(background, pt1=(pixValues[pen][1][0]-numOfSquaresFunc(square,ImgW,ImgH)[2],pixValues[pen][1][1]-numOfSquaresFunc(square,ImgW,ImgH)[2]), 
                  pt2=(pixValues[pen][1][0]+numOfSquaresFunc(square,ImgW,ImgH)[2],pixValues[pen][1][1]+numOfSquaresFunc(square,ImgW,ImgH)[2]), 
                  color=(pixValues[pen][0][0],pixValues[pen][0][1],pixValues[pen][0][2]), thickness=-1)
  
  cv2.imwrite(f"{os.getcwd()}/pixed/"+ file, cv2.cvtColor(background, cv2.COLOR_RGB2BGR,background))
