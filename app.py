import cv2
import torch
import numpy as np
import gradio as gr
import paddlehub as hub
from PIL import Image
from methods.img2pixl import pixL
from examples.pixelArt.combine import combine

model = hub.Module(name='U2Net')
device = "cuda" if torch.cuda.is_available() else "cpu"
face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device, size=512)
model = torch.hub.load("bryandlee/animegan2-pytorch", "generator", device=device).eval()

def initilize(image,pixel_size, checkbox1):
    image = Image.open(image.name).convert("RGB")
    image = cv2.cvtColor(np.asarray(face2paint(model, image)), cv2.COLOR_BGR2RGB)

    if checkbox1:#With the Object Oriented Inference
      result = model.Segmentation(
          images=[image],
          paths=None,
          batch_size=1,
          input_size=320,  
          output_dir='output',
          visualization=True)
      result = combine().combiner(images = pixL().toThePixL([result[0]['front'][:,:,::-1], result[0]['mask']], 
                                                        pixel_size),
                                background_image = image)
    else: #Without the Object Oriented Inference
      result = pixL().toThePixL([image], pixel_size)
    return result

inputs = [gr.inputs.Image(type='file', label="Image"),
               gr.Slider(4, 100, value=12, step = 2, label="Pixel Size"),
               gr.Checkbox(label="Object-Oriented Inference", value=False)]
outputs = [gr.Image(type="file",label="Front")]

description = "Pixera for GIF and Video will be released soon."
gr.Interface(fn = initilize,
                    inputs = inputs,
                    outputs = outputs,
                    description=description).launch()

