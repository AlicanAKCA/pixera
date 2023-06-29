import os
import cv2
import torch
import random
import numpy as np
import gradio as gr
from util import util
from util.img2pixl import pixL
from data import create_dataset
from models import create_model
from options.test_options import TestOptions

opt = TestOptions().parse()
opt.num_threads = 0
opt.batch_size = 1
opt.display_id = -1
opt.no_dropout = True

model = create_model(opt)
model.setup(opt)

num_inferences = 0

def preprocess(image):

  im_type = None
  imgH, imgW = image.shape[:2]
  aspect_ratio = imgW / imgH


  if 0.75 <= aspect_ratio <= 1.75:

    image = cv2.resize(image, (512, 512))
    image = pixL().toThePixL(image,6,False)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.asarray([image])
    image = np.transpose(image, (0, 3, 1, 2))

    image = inference(image)

    return image

  elif 1.75 <= aspect_ratio: # upper boundary

    image = cv2.resize(image, (1024, 512))
    middlePoint = image.shape[1] // 2
    half_1 = image[:,:middlePoint]
    half_2 = image[:,middlePoint:]
    images = [half_1,half_2]
      
    for image in images:
      image = pixL().toThePixL(image,6,False)
      image = np.asarray([image])
      image = np.transpose(image, (0, 3, 1, 2))
      image = inference(image)
    
    image = cv2.hconcat([images[0], images[1]])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

  elif 0.00 <= aspect_ratio <= 0.75:

    image = cv2.resize(image, (512, 1024))
    middlePoint = image.shape[0] // 2
    half_1 = image[:middlePoint,:]
    half_2 = image[middlePoint:,:]
    images = [half_1,half_2]
      
    for image in images:
      image = pixL().toThePixL(image,6,False)
      image = np.asarray([image])
      image = np.transpose(image, (0, 3, 1, 2))
      image = inference(image)
    
    image = cv2.vconcat([images[0], images[1]])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def postprocess(image):
  image = util.tensor2im(image)
  return image

def inference(image):
  global model

  data = {"A": None, "A_paths": None}
  data['A'] = torch.FloatTensor(image)

  model.set_input(data)
  model.test()

  image = model.get_current_visuals()['fake']

  return image

def pixera_CYCLEGAN(image):
  global num_inferences
  
  image = preprocess(image)

  image = postprocess(image)

  num_inferences += 1
  print(num_inferences)
  
  return image

title_ = "Pixera: Create your own Pixel Art"
description_ = ""

examples_path = f"{os.getcwd()}/imgs"
examples_ = os.listdir(examples_path)
random.shuffle(examples_)
examples_ = [[f"{examples_path}/{example}"] for example in examples_]

 
demo = gr.Interface(pixera_CYCLEGAN, inputs = [gr.Image(show_label= False)],
                                     outputs = [gr.Image(show_label= False)], 
                                     examples = examples_,
                                     title = title_,
                                     description= description_)
demo.launch(debug= True, share=True)
