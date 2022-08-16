import cv2

def split(fname):
    media = cv2.VideoCapture(fname)
    frames = []
    while True:
        ret, cv2Image = media.read()
        if not ret:
            break
        frames.append(cv2Image)
    return frames

split("test.gif")