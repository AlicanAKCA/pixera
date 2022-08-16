import os
import cv2
import torch
import warnings
import numpy as np
import gradio as gr
import paddlehub as hub
from PIL import Image
from methods.img2pixl import pixL
from examples.pixelArt.combine import combine
from methods.media import Media

warnings.filterwarnings("ignore")

U2Net = hub.Module(name='U2Net')
device = "cuda" if torch.cuda.is_available() else "cpu"
face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device, size=512)
model = torch.hub.load("bryandlee/animegan2-pytorch", "generator", device=device).eval()


def initilize(media,pixel_size,checkbox1):
    #Author:  Alican Akca
    if media.name.endswith('.gif'):
      return Media().split(media.name,pixel_size)
    elif media.name.endswith('.mp4'):
      return Media().split(media.name,pixel_size)
    else:
      media = Image.open(media.name).convert("RGB")
      media = cv2.cvtColor(np.asarray(face2paint(model, media)), cv2.COLOR_BGR2RGB)
      if checkbox1:
        result = U2Net.Segmentation(images=[media],
                                    paths=None,
                                    batch_size=1,
                                    input_size=320,  
                                    output_dir='output',
                                    visualization=True)
        result = combine.combiner(images = pixL().toThePixL([result[0]['front'][:,:,::-1], result[0]['mask']], 
                                                        pixel_size),
                                background_image = media)
      else:
        result = pixL().toThePixL([media], pixel_size)
      result = Image.fromarray(result)
      result.save('cache.png')
      return [None, result, 'cache.png']

inputs = [gr.File(label="Media"),
               gr.Slider(4, 100, value=12, step = 2, label="Pixel Size"),
               gr.Checkbox(label="Object-Oriented Inference", value=False)]
outputs = [gr.Video(label="Pixed Media"),
           gr.Image(label="Pixed Media"),
           gr.File(label="Download")]

title = "Pixera: Create your own Pixel Art"
description = """Mobile applications will have released soon ^^ """

gr.Interface(fn = initilize,
                    inputs = inputs,
                    outputs = outputs,
                    title=title,
                    description=description).launch()

