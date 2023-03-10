import cv2
import random
import numpy as np
from PIL import Image
#Author: Alican Akca

class pixL:

  def __init__(self,numOfSquaresW = None, numOfSquaresH= None, size = [True, (512,512)],square = 6,ImgH = None,ImgW = None,images = [],background_image = None):
    self.images = images
    self.size = size
    self.ImgH = ImgH
    self.ImgW = ImgW
    self.square = square
    self.numOfSquaresW = numOfSquaresW
    self.numOfSquaresH = numOfSquaresH

  def preprocess(self):
    for image in self.images:

      size = (image.shape[0] - (image.shape[0] % 4), image.shape[1] - (image.shape[1] % 4))
      image = cv2.resize(image, size)
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

  def optimizer(RGB):
    
    R_ = RGB[2]
    G_ = RGB[1]
    B_ = RGB[0]

    if R_ < 50 and G_ < 50 and B_ < 50: 
      
      return (R_, G_, B_)

    else:
      sign = lambda x, y: random.choice([x,y])

      R_ = RGB[2] + sign(+1,-1)*random.randint(1,10)
      G_ = RGB[1] + sign(+1,-1)*random.randint(1,10)
      B_ = RGB[0] + sign(+1,-1)*random.randint(1,10)

      R_ = 0 if R_ < 0 else (255 if R_ > 255 else R_)
      G_ = 0 if G_ < 0 else (255 if G_ > 255 else G_)
      B_ = 0 if B_ < 0 else (255 if B_ > 255 else B_)

      return (R_, G_, B_)

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
                    pt1=(pixValues[pen][1][0] - self.square, pixValues[pen][1][1] - self.square), #0, 0 -> 0, 0
                    pt2=(pixValues[pen][1][0], pixValues[pen][1][1]), #6, 6 -> 3, 3
                    color=(pixL.optimizer(pixValues[pen][0])), 
                    thickness=-1)
      
      cv2.rectangle(background, 
                    pt1=(pixValues[pen][1][0], pixValues[pen][1][1] - self.square), #0, 0 -> 3, 0
                    pt2=(pixValues[pen][1][0] + self.square, pixValues[pen][1][1]), #6, 6 -> 6, 3
                    color=(pixL.optimizer(pixValues[pen][0])), 
                    thickness=-1)
      
      cv2.rectangle(background, 
                    pt1=(pixValues[pen][1][0] - self.square, pixValues[pen][1][1]), #0, 0 -> 0, 3
                    pt2=(pixValues[pen][1][0], pixValues[pen][1][1] + self.square), #6, 6 -> 3, 6
                    color=(pixL.optimizer(pixValues[pen][0])), 
                    thickness=-1)
      
      cv2.rectangle(background, 
                    pt1=(pixValues[pen][1][0], pixValues[pen][1][1]), #0, 0 -> 3, 3
                    pt2=(pixValues[pen][1][0] + self.square, pixValues[pen][1][1] + self.square), #6, 6 -> 6, 6
                    color=(pixL.optimizer(pixValues[pen][0])), 
                    thickness=-1)
      
    background = np.array(background).astype(np.uint8)
    background = cv2.resize(background, (self.ImgW,self.ImgH), interpolation = cv2.INTER_AREA)
    
    return background
