from keras.models import model_from_json
with open('model.json','r') as f:
    model = model_from_json(f.read())

model.load_weights('model.h5')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from math import ceil

print("Reading input from the file imeages")

def pre(input_path = 'images',output_path='output'):
    name=['Cat','Dog']
    lis = os.listdir(input_path)
    images = len(lis)

    counter=[0,0]
    
    for i in range(images):
        print('image : ',lis[i])
        image = cv2.resize(cv2.imread('images/'+lis[i]),(128,128))/255.0
        pred = int(round(model.predict(np.array([image]))[0][0]))
        counter[pred]+=1
        os.rename(input_path+"/"+lis[i],output_path+"/"+name[pred]+"_" + str(counter[pred]) + ".jpg")


pre()
print("Done successfully")