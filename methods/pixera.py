#Author: Alican Akca

import os
import cv2
import numpy as np
from PIL import Image, ImageOps

class pixL:

  def __self__(self,numOfSquaresW, numOfSquaresH, size = (400,400),square = 4,ImgH = None,ImgW = None,file = None,img = None):

    self.img = img
    self.size = size
    self.ImgH = ImgH
    self.ImgW = ImgW
    self.file = file 
    self.square = square
    self.numOfSquaresW = numOfSquaresW
    self.numOfSquaresH = numOfSquaresH
    self.path = f"{os.getcwd()}/dataset"

  def toThePixL(self):
    files = os.listdir(self.path)
    for self.file in files:
      img = Image.open(self.path)
      img = img.convert("RGB")
      self.img = img.resize(self.size)
      self.ImgW, self.ImgH = self.img.size
      pixL.epicAlgorithm(self.square,self.ImgW,self.ImgH,self.file,self.img)

  def numOfSquaresFunc(self):
    self.numOfSquaresW = round((self.ImgW / self.square) + 1)
    self.numOfSquaresH = round((self.ImgH / self.square) + 1)
    return (self.numOfSquaresW,self.numOfSquaresH) , self.square

  def epicAlgorithm(self):
    pixValues = []
    pixL.numOfSquaresFunc()
    
    for j in range(1,self.numOfSquaresH):

      for i in range(1,self.numOfSquaresW):
        
        pixValues.append((self.img.getpixel((
              i * self.square - self.square//2,
              j * self.square - self.square//2)),
              (i * self.square - self.square//2,
              j * self.square - self.square//2)))
    
    background = 255 * np.ones(shape=[self.ImgH - self.square, 
                                      self.ImgW - self.square*2, 3], 
                                      dtype=np.uint8)                
    
    for pen in range(len(pixValues)):

      cv2.rectangle(background, 
                    pt1=(pixValues[pen][1][0] - self.square,pixValues[pen][1][1] - self.square), 
                    pt2=(pixValues[pen][1][0] + self.square,pixValues[pen][1][1] + self.square), 
                    color=(pixValues[pen][0][0],pixValues[pen][0][1],pixValues[pen][0][2]), 
                    thickness=-1)

    
    cv2.imwrite(f"{os.getcwd()}/output/{self.file}", 
                cv2.cvtColor(background, cv2.COLOR_RGB2BGR,background))

class Augmentation:
  pass
