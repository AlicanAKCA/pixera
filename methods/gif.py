import cv2
import numpy as np
from PIL import Image
from methods.img2pixl import pixL
from examples.pixelArt.combine import combine
from examples.pixelArt.white_box_cartoonizer.cartoonize import WB_Cartoonize
pixl = pixL()
combine = combine()
from PIL import Image

def process_image(filename):
    gif = Image.open(filename)
    frames = []
    for i in range(gif.n_frames):
        gif.seek(i)
        frame = Image.new('RGB', gif.size)
        frame.paste(gif)
        frame = np.array(frame)
        frames.append(frame)
    result = pixl.toThePixL(frames, 6)
    frames = []
    for frame in result:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)    
        frames.append(frame)
    frames[0].save('new.gif', append_images=frames[1:], save_all=True, loop=1)


if __name__ == '__main__':
    process_image('giphy.gif')