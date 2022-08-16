import cv2
import torch
import imageio
from methods.img2pixl import pixL


device = "cuda" if torch.cuda.is_available() else "cpu"
face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device, size=512)
model = torch.hub.load("bryandlee/animegan2-pytorch", "generator", device=device).eval()

class Media:
    #Author:  Alican Akca
    def __init__(self,fname = None,pixel_size = None):
        self.fname = fname
        self.pixel_size = pixel_size

    def split(self,fname,pixel_size):
        media = cv2.VideoCapture(fname)
        frames = []
        while True:
            ret, cv2Image = media.read()
            if not ret:
                break
            frames.append(cv2Image)
        frames = pixL().toThePixL(frames, pixel_size)
        imageio.mimsave('cache.gif', frames)
        output_file = "cache.mp4"
        out = cv2.VideoWriter(output_file,cv2.VideoWriter_fourcc(*'h264'), 15, (frames[0].shape[1],frames[0].shape[0]))
        for i in range(len(frames)):
            out.write(frames[i])
        out.release()
        return output_file
