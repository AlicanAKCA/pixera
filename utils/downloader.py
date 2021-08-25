import os
from fastai.vision import *
import PIL 

def downloader():
  path = Path(os.getcwd() + '/dataset/')
  folder = 'img'
  file = os.getcwd() +'utils/dataset.csv'
  mainDest = path/folder
  mainDest.mkdir(parents=True, exist_ok=True)
  download_images(path/file, mainDest, max_pics=5000, max_workers=0)
  files = os.listdir(str(mainDest))  
  for index, file in enumerate(files):
    os.rename(os.path.join(str(mainDest), file), os.path.join(str(mainDest), ''.join([str(index+1), '.jpg'])))



downloader()

#burası olmuyo