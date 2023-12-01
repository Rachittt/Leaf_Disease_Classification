import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
from PIL import Image



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts","training", "model.h5"))
        imagename = self.filename

        image = Image.open(imagename)
        image = image.resize((224,224))
        img = np.asarray(image)
        img_array = tf.expand_dims(img, 0)

        class_names = ['Pepper__bell___Bacterial_spot',
                        'Pepper__bell___healthy',
                        'Potato___Early_blight',
                        'Potato___Late_blight',
                        'Potato___healthy',
                        'Tomato_Bacterial_spot',
                        'Tomato_Early_blight',
                        'Tomato_Late_blight',
                        'Tomato_Leaf_Mold',
                        'Tomato_Septoria_leaf_spot',
                        'Tomato_Spider_mites_Two_spotted_spider_mite',
                        'Tomato__Target_Spot',
                        'Tomato__Tomato_YellowLeaf__Curl_Virus',
                        'Tomato__Tomato_mosaic_virus',
                        'Tomato_healthy']

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        print(predicted_class)
        return [{"image": predicted_class}]