import argparse
import os
from scipy import misc
from PIL import Image
import imageio
import numpy as np
from read_params import read_params

def extract_bayer_channels(raw):
    # Reshape the input bayer image

    ch_B = raw[1::2, 1::2]
    ch_R = raw[1::2, 0::2]
    ch_Gb = raw[0::2, 1::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_R, ch_Gb, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4*255)

    return RAW_norm 



def load_testing_batch(test_path_c, test_path_r, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE):
    # Loading test images
    NUM_TESTING_IMAGES = len([name for name in os.listdir(test_path_r) if os.path.isfile(os.path.join(test_path_r, name))])

    test_data = np.zeros((NUM_TESTING_IMAGES, PATCH_WIDTH, PATCH_HEIGHT, 4))
    test_target = np.zeros((NUM_TESTING_IMAGES, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3))

    for i in range(0,NUM_TESTING_IMAGES):
        inp_img = np.asarray(imageio.imread((test_path_r + str(i) + '.png')))    
        inp_img = extract_bayer_channels(inp_img)
        test_data[i, :] = inp_img

        inp_img = np.asarray(Image.open(test_path_c + str(i) + '.jpg'))

        resize_h = int(inp_img.shape[0] * (DSLR_SCALE/2))
        resize_w = int(inp_img.shape[1] * (DSLR_SCALE/2))

        inp_img = np.array(Image.fromarray(inp_img).resize((resize_w,resize_h), resample=Image.BICUBIC))
        inp_img = np.float16(np.reshape(inp_img, [1, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3])) / 255
        test_target[i, :] = inp_img
    
    return test_data, test_target



def load_training_batch(train_path_c, train_path_r, TRAIN_SIZE, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE):
    # Load training data
    NUM_TRAINING_IMAGES = len([name for name in os.listdir(train_path_r) if os.path.isfile(os.path.join(train_path_r, name))])

    TRAIN_IMAGES = np.random.choice(np.arange(0, NUM_TRAINING_IMAGES), TRAIN_SIZE, replace=False)

    train_data = np.zeros((TRAIN_SIZE, PATCH_WIDTH, PATCH_HEIGHT, 4))
    train_target = np.zeros((TRAIN_SIZE, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3))

    i=0
    for img in TRAIN_IMAGES:
        inp_img = np.asarray(imageio.imread((train_path_r + str(img) + '.png')))
        inp_img = extract_bayer_channels(inp_img)
        train_data[i, :] = inp_img

        inp_img = np.asarray(Image.open(train_path_c + str(img) + '.jpg'))

        resize_h = int(inp_img.shape[0] * (DSLR_SCALE/2))
        resize_w = int(inp_img.shape[1] * (DSLR_SCALE/2))

        inp_img = np.array(Image.fromarray(inp_img).resize((resize_w,resize_h), resample=Image.BICUBIC))
        inp_img = np.float16(np.reshape(inp_img, [1, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3])) / 255
        train_target[i, :] = inp_img

        i += 1

    return train_data, train_target

def get_data(config_path, TRAIN_SIZE, DSLR_SCALE):
    config = read_params(config_path)
    train_path_c = config['data_path']['train']['train_canon']
    train_path_r = config['data_path']['train']['train_raw']

    test_path_c = config['data_path']['test']['test_canon']
    test_path_r = config['data_path']['test']['test_raw']
    test_path_full = config['data_path']['test']['test_full']
    test_path_v = config['data_path']['test']['test_visualized']

    PATCH_WIDTH = config['data_load']['PATCH_WIDTH']
    PATCH_HEIGHT = config['data_load']['PATCH_HEIGHT']

    train_data, train_target = load_training_batch(train_path_c, train_path_r, TRAIN_SIZE ,PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)
    test_data, test_target = load_testing_batch(test_path_c, test_path_r, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)

    return train_data, train_target, test_data, test_target


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml', help='params file')
    parsed_args = args.parse_args()
    train_dataset, train_target, test_dataset, test_target = get_data(config_path=parsed_args.config, TRAIN_SIZE=2, DSLR_SCALE=8)
    print(train_dataset.shape)
    print(train_target.shape)
    print(test_dataset.shape)
    print(test_target.shape)