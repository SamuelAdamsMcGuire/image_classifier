import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input #double check
from PIL import Image
import cv2
import os

def process(frame):
    """
    process pic for capture script
    """
    # resize the image
    img = cv2.resize(frame, (224, 224))
    #img = img.resize((224, 224))
        
    # convert color from BGR to RGB   
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB)#how to change to black/white for MNist dataset?
   
    
    # possibly do some preprocessing e.g. converting it to range [0,1]
    # reshape it into shape (1, target_width, target_height, 3)
    img = preprocess_input(img) #change to densenet!!!
    img = img.reshape(1, 224, 224, 3)
    return img
   
  
if __name__ == "__main__":

    pic = process('2020-04-08-22-19-27.png')
    print(pic)
    print(pic.shape)

    


