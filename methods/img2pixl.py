import cv2
import numpy as np
from PIL import Image

class pixL:
  #Author:  Alican Akca
  def __init__(self,numOfSquaresW = None, numOfSquaresH= None, size = [False, (512,512)],square = 6,ImgH = None,ImgW = None,images = [],background_image = None):
    self.images = images
    self.size = size
    self.ImgH = ImgH
    self.ImgW = ImgW
    self.square = square
    self.numOfSquaresW = numOfSquaresW
    self.numOfSquaresH = numOfSquaresH

  def preprocess(self):
    for image in self.images:

      size = (self.ImgW - (self.ImgW % 4), self.ImgH - (self.ImgH % 4))
      image = cv2.resize(image, size)
      image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    if len(self.images) == 1:
      return self.images[0]
    else:
      return self.images


  def toThePixL(self,images, pixel_size):
    self.images = []
    self.square = pixel_size
    for image in images:
      image = Image.fromarray(image)
      image = image.convert("RGB")
      self.ImgW, self.ImgH = image.size
      self.images.append(pixL.epicAlgorithm(self, image))
    return pixL.preprocess(self)

  def numOfSquaresFunc(self):
    self.numOfSquaresW = round((self.ImgW / self.square) + 1)
    self.numOfSquaresH = round((self.ImgH / self.square) + 1)

  def epicAlgorithm(self, image):
    pixValues = []
    pixL.numOfSquaresFunc(self)

    for j in range(1,self.numOfSquaresH):

      for i in range(1,self.numOfSquaresW):
        
        pixValues.append((image.getpixel((
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
                    color=(pixValues[pen][0][2],pixValues[pen][0][1],pixValues[pen][0][0]), 
                    thickness=-1)
    background = np.array(background).astype(np.uint8)
    background = cv2.resize(background, (self.ImgW,self.ImgH), interpolation = cv2.INTER_AREA)

    return background