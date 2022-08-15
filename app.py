import cv2
import torch
import numpy as np
import gradio as gr
import paddlehub as hub
from PIL import Image
from methods.img2pixl import pixL
from examples.pixelArt.combine import combine
#cartoonlama ve ooi frame frame çalışması için büyük ihtimal image2pix e taşınacak
model = hub.Module(name='U2Net')
device = "cuda" if torch.cuda.is_available() else "cpu"
face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device, size=512)
model = torch.hub.load("bryandlee/animegan2-pytorch", "generator", device=device).eval()

def GIF(fname,pixel_size):
    gif = Image.open(fname)
    frames = [] 
    for i in range(gif.n_frames): #First Step: Splitting the GIF into frames
        gif.seek(i)
        frame = Image.new('RGB', gif.size)
        frame.paste(gif)
        frame = np.array(frame)
        frames.append(frame)
    result = pixL().toThePixL(frames, pixel_size)
    for frame in result:          #Second Step: Adding Cartoon Effect to each frame
        frame = Image.fromarray(frame)
        frame = cv2.cvtColor(np.asarray(face2paint(model, frame)), cv2.COLOR_BGR2RGB)
    frames = []
    for frame in result:          #Third Step: Combining the frames into a GIF
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame = Image.fromarray(frame)    
      frames.append(frame)
    frames[0].save('cache.gif', append_images=frames, save_all=True, loop=1)
    cache = Image.open('cache.gif')
    return cache

def initilize(image,pixel_size,checkbox1):
    if image.name.endswith('.gif'):
      GIF(image.name,pixel_size)
    else:
      image = Image.open(image.name).convert("RGB")
      image = cv2.cvtColor(np.asarray(face2paint(model, image)), cv2.COLOR_BGR2RGB)
      if checkbox1:
        result = model.Segmentation(
        images=[image],
        paths=None,
        batch_size=1,
        input_size=320,  
        output_dir='output',
        visualization=True)
        result = combine.combiner(images = pixL().toThePixL([result[0]['front'][:,:,::-1], result[0]['mask']], 
                                                        pixel_size),
                                background_image = image)
      else:
        result = pixL().toThePixL([image], pixel_size)
      return Image.fromarray(result)

inputs = ["file",
               gr.Slider(4, 100, value=12, step = 2, label="Pixel Size"),
               gr.Checkbox(label="Object-Oriented Inference", value=False)]
outputs = [gr.Image(type="pil",label="Front")]

gr.Interface(fn = initilize,
                    inputs = inputs,
                    outputs = outputs).launch()

