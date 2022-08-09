import cv2
import time
import shutil
import numpy as np
import gradio as gr
from PIL import Image
import paddlehub as hub
from methods.img2pixl import pixL
from examples.pixelArt.combine import combine
from examples.pixelArt.white_box_cartoonizer.cartoonize import WB_Cartoonize
model = hub.Module(name='U2Net')
pixl = pixL()
combine = combine()

def GIF(image,pixel_size):
    fname = image.name
    time.sleep(10)
    shutil.copy(fname, 'temp.gif')
    time.sleep(10)
    gif = Image.open("temp.gif")
    
    print(fname, gif.n_frames)
    frames = []
    for i in range(gif.n_frames):
        gif.seek(i)
        frame = Image.new('RGBA', gif.size)
        frame.paste(gif)
        frame = np.array(frame)
        frames.append(frame)
    print(len(frames))
    
    result = pixl.toThePixL(frames, pixel_size)
    print(len(result), result[0].shape)
    frames = []
    for frame in result:
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)    
        frames.append(frame)
    print(type(frames), len(frames), type(frames[0]), frames[0].size)
    frames[0].save('new.gif', append_images=frames, save_all=True, loop=1)
    return Image.open('cache.gif')

def func_tab1(image,pixel_size, checkbox1):
  if image.name.endswith('.gif'):
    print(type(image),type(image.name))
    GIF(image,pixel_size)
  else:
    image = cv2.imread(image.name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = WB_Cartoonize().infer(image)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if checkbox1:
      result = model.Segmentation(
          images=[image],
          paths=None,
          batch_size=1,
          input_size=320,  
          output_dir='output',
          visualization=True)
      result = combine.combiner(images = pixl.toThePixL([result[0]['front'][:,:,::-1], result[0]['mask']], 
                                                        pixel_size),
                                background_image = image)
    else:
      result = pixl.toThePixL([image], pixel_size)
    return result

inputs_tab1 = [gr.inputs.Image(type='file', label="Image"),
               gr.Slider(4, 100, value=12, step = 2, label="Pixel Size"),
               gr.Checkbox(label="Object-Oriented Inference", value=False)]
outputs_tab1 = [gr.Image(type="file",label="Front")]

gr.Interface(fn = func_tab1,
                    inputs = inputs_tab1,
                    outputs = outputs_tab1).launch()

