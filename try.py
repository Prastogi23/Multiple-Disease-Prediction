import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
savedModel=load_model('gfgModel.h5')
savedModel.summary()
image_path = "1062_right.jpg"
image_size=224
image = cv2.imread(image_path,cv2.IMREAD_COLOR) #load 
image = cv2.resize(image,(image_size,image_size)) #resize
arr = np.array(image)
x = np.array(arr).reshape(-1,image_size,image_size,3)
y = savedModel.predict([x])
print(y)
if y[0] >= 0.5:
    ans = 1
else:
    ans = 0
print(ans)