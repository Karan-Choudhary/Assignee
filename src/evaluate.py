import yaml
import os
from read_params import read_params
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
from display.output import generate_images
from get_data import get_data

def evaluate_model(config_path):
    config = read_params(config_path)
    SAMPLES = config['test']['num_samples']
    MODEL_DIR = config['model_dir']
    RESULTS = config['result']
    MODEL_NUM = config['model_num']

    DSLR_SCALE = float(1)/(2**((MODEL_NUM+1) - 1))

    test_data, test_target = get_data(config_path, SAMPLES, DSLR_SCALE, False, True)

    for input_image, target_image in zip(test_data, test_target):
        generate_images(input_image, target_image, RESULTS)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml', help='params file')
    parsed_args = args.parse_args()
    evaluate_model(config_path=parsed_args.config)