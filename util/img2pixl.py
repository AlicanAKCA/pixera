import cv2
import random
import numpy as np
from PIL import Image
#Author: Alican Akca

class pixL:

  def __init__(self,numOfSquaresW = None, numOfSquaresH= None, size = [True, (512,512)],square = 6,ImgH = None,ImgW = None,image = None,background = None, pixValues = []):
    self.size = size
    self.ImgH = ImgH
    self.ImgW = ImgW
    self.image = image
    self.square = square
    self.pixValues = pixValues
    self.background = background
    self.numOfSquaresW = numOfSquaresW
    self.numOfSquaresH = numOfSquaresH

  def toThePixL(self,image, pixel_size, segMode= False):
    self.square = pixel_size
    self.image = Image.fromarray(image).convert("RGB").resize((512,512))
    self.ImgW, self.ImgH = self.image.size
    self.image = pixL.colorPicker(self)
    pixL.complier(self)
    if segMode == True:
      return pixL.postprocess(self), self.pixValues
    else:
      return pixL.postprocess(self)

  def postprocess(self):
    image = self.background
    size = (image.shape[0] - (image.shape[0] % 4), image.shape[1] - (image.shape[1] % 4))
    image = cv2.resize(image, size)
    return image

  def numOfSquaresFunc(self):
    self.numOfSquaresW = round((self.ImgW / self.square) + 1)
    self.numOfSquaresH = round((self.ImgH / self.square) + 1)

  def optimizer(RGB):

    R_ = RGB[2]
    G_ = RGB[1]
    B_ = RGB[0]

    if R_ < 50 and G_ < 50 and B_ < 50:

      return (R_, G_, B_)

    elif 220 < R_ < 255 and 220 < G_ < 255 and 220 < B_ < 255:

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

  def colorPicker(self):
    pixL.numOfSquaresFunc(self)

    for j in range(1,self.numOfSquaresH):

      for i in range(1,self.numOfSquaresW):

        self.pixValues.append((self.image.getpixel((
              i * self.square - self.square//2,
              j * self.square - self.square//2)),
              (i * self.square - self.square//2,
              j * self.square - self.square//2)))

    self.background = 255 * np.ones(shape=[self.ImgH - self.square,
                                           self.ImgW - self.square*2, 3],
                                    dtype=np.uint8)

  def PEN(self,coorX,coorY,R,G,B):
    SQUARE = self.square
    cv2.rectangle(self.background,
                 pt1=(coorX - SQUARE, coorY - SQUARE), #0, 0 -> 0, 0
                 pt2=(coorX, coorY), #6, 6 -> 3, 3
                 color=(pixL.optimizer((R,G,B))),
                 thickness=-1)

    cv2.rectangle(self.background,
                 pt1=(coorX, coorY - SQUARE), #0, 0 -> 3, 0
                 pt2=(coorX + SQUARE, coorY), #6, 6 -> 6, 3
                 color=(pixL.optimizer((R,G,B))),
                 thickness=-1)

    cv2.rectangle(self.background,
                  pt1=(coorX - SQUARE, coorY), #0, 0 -> 0, 3
                  pt2=(coorX, coorY + SQUARE), #6, 6 -> 3, 6
                  color=(pixL.optimizer((R,G,B))),
                  thickness=-1)

    cv2.rectangle(self.background,
                 pt1=(coorX, coorY), #0, 0 -> 3, 3
                 pt2=(coorX + SQUARE, coorY + SQUARE), #6, 6 -> 6, 6
                 color=(pixL.optimizer((R,G,B))),
                 thickness=-1)

  def complier(self):
    for index, value in enumerate(self.pixValues):
      (R,G,B), (coorX, coorY) = value
      pixL.PEN(self,coorX,coorY,R,G,B)
    self.background = np.array(self.background).astype(np.uint8)
    self.background = cv2.resize(self.background, (self.ImgW,self.ImgH), interpolation = cv2.INTER_AREA)
