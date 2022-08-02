import cv2
import numpy as np
import gradio as gr
import paddlehub as hub
from methods.img2pixl import pixL
from examples.pixelArt.combine import combine
from examples.pixelArt.white_box_cartoonizer.cartoonize import WB_Cartoonize
model = hub.Module(name='U2Net')
pixl = pixL()
combine = combine()

def func_tab1(image,pixel_size, checkbox1):
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

def func_tab2():
  pass

inputs_tab1 = [gr.inputs.Image(type='file', label="Image"),
               gr.Slider(4, 100, value=12, step = 2, label="Pixel Size"),
               gr.Checkbox(label="Object-Oriented Inference", value=False)]
outputs_tab1 = [gr.Image(type="numpy",label="Front")]

inputs_tab2 = [gr.Video()]
outputs_tab2 = [gr.Video()]

tab1 = gr.Interface(fn = func_tab1,
                    inputs = inputs_tab1,
                    outputs = outputs_tab1)
#Pixera for Videos
tab2 = gr.Interface(fn = func_tab2,
                    inputs = inputs_tab2,
                    outputs = outputs_tab2)

gr.TabbedInterface([tab1], ["Pixera for Images"]).launch()

