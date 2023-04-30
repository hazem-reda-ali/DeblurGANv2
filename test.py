"""
print("Running test.py")#check that testt.py is running
import torch
print()
"""







import argparse
import os

import cv2
import numpy as np
import tensorflow as tf

from models.networks import FPN
from utils.config import process_config
from utils.utils import get_args
from utils.image_utils import load_image, save_image

def test_model(config, model_path, test_data_path, result_path):
    # Load model
    model = FPN(config)

    # Load saved weights
    model.load_weights(model_path)

    # Load test image
    test_image = load_image(test_data_path)

    # Predict deblurred image
    predicted_image = model.predict(np.expand_dims(test_image, axis=0))[0]

    # Save result
    save_image(predicted_image, result_path)

if __name__ == '__main__':
    # Parse command-line arguments
    args = get_args()

    # Load configuration file
    config = process_config(args.config)

    # Test the model
    test_model(config, args.model_path, args.test_data_path, args.result_path)


"""
import cv2
import numpy as np
from tensorflow.keras.models import load_model

import keras; print(keras.__version__)
import tensorflow as tf; print(tf.__version__)

#model_path = 'C:/Users/hazem/OneDrive/Desktop/Bachelor/Code/best_fpn.h5' 
#model_path = 'C:/Users/hazem/OneDrive/Desktop/test/best_fpn.h5'
model_path = 'C:/Users/hazem/OneDrive/Desktop/test/fpn_inception.h5'
model = load_model(model_path)


test_image_path = 'C:/Users/hazem/OneDrive/Documents/GitHub/DeblurGANv2/test_img/000027.png'
test_image = cv2.imread(test_image_path)


cv2.imshow('Input Image', test_image)
cv2.imshow('Output Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
import os
os.chdir('C:/Users/hazem/OneDrive/Desktop/test')

import h5py

with h5py.File('fpn_inception.h5', 'r') as f:
    for key in f.keys():
        print(key)
        
        """









"""
import h5py

# open the h5 file
with h5py.File(r'C:\Users\hazem\OneDrive\Desktop\test\fpn_mobilenet.h5', 'r') as f:
    # print the attributes of the file
    print(list(f.attrs.keys()))
      """#testing if it sees .h5 file