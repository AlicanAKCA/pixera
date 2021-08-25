import os
import cv2
import numpy as np
import tensorflow as tf 

ImgPath = os.getcwd() + "/dataset/img"
ImgIds = os.listdir(ImgPath)
Imgs = []
iterations = 50
vector_noise_dim = 180
count_example = 16
batch_size=12
count_buffer = 60000

for image in ImgIds:
  img = cv2.imread(f"{ImgPath}/{image}")
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  Imgs.append(img)

X_Train = np.asarray(Imgs)
X_Train = X_Train / 255.
Train_Data = tf.data.Dataset.from_tensor_slices(X_Train).shuffle(count_buffer).batch(batch_size)

print(len(X_train))