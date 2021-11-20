#Author: Alican Akca

import os
import cv2
import numpy as np
from PIL import Image

def toThePixL(square = 16):
  files = os.listdir(f"{os.getcwd()}/dataset/original")
  for index, file in enumerate(files):
    img = Image.open(f"{os.getcwd()}/dataset/original/{file}")
    img = img.convert("RGB")
    ImgW, ImgH = img.size
    epicAlgorithm(square,ImgW,ImgH,file,img)

def numOfSquaresFunc(square: int,ImgW: int,ImgH: int):
  numOfSquaresW = round((ImgW / square) - 1)
  numOfSquaresH = round((ImgH / square) - 1)
  return (numOfSquaresW,numOfSquaresH) , square, square//2

def epicAlgorithm(square: int,ImgW: int,ImgH: int,file,img):
  pixValues = []

  for j in range(1,numOfSquaresFunc(square,ImgW,ImgH)[0][1]+1):

    for i in range(1,numOfSquaresFunc(square,ImgW,ImgH)[0][0]+1):
      
      pixValues.append((img.getpixel((
            i * numOfSquaresFunc(square,ImgW,ImgH)[1] - numOfSquaresFunc(square,ImgW,ImgH)[2],
            j * numOfSquaresFunc(square,ImgW,ImgH)[1]- numOfSquaresFunc(square,ImgW,ImgH)[2])),
            (i * numOfSquaresFunc(square,ImgW,ImgH)[1] - numOfSquaresFunc(square,ImgW,ImgH)[2],
            j * numOfSquaresFunc(square,ImgW,ImgH)[1] - numOfSquaresFunc(square,ImgW,ImgH)[2])))
 
  background = 255 * np.ones(shape=[ImgH - numOfSquaresFunc(square,ImgW,ImgH)[1], 
                                    ImgW - numOfSquaresFunc(square,ImgW,ImgH)[1]*2, 3], 
                                    dtype=np.uint8)         
  
  for pen in range(len(pixValues)):
    
    cv2.rectangle(background, 
                  pt1=(pixValues[pen][1][0]-numOfSquaresFunc(square,ImgW,ImgH)[2],pixValues[pen][1][1]-numOfSquaresFunc(square,ImgW,ImgH)[2]), 
                  pt2=(pixValues[pen][1][0]+numOfSquaresFunc(square,ImgW,ImgH)[2],pixValues[pen][1][1]+numOfSquaresFunc(square,ImgW,ImgH)[2]), 
                  color=(pixValues[pen][0][0],pixValues[pen][0][1],pixValues[pen][0][2]), 
                  thickness=-1)
  
  cv2.imwrite(f"{os.getcwd()}/dataset/pixed/{file}", 
              cv2.cvtColor(background, cv2.COLOR_RGB2BGR,background))
