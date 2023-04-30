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