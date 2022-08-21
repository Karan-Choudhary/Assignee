import tensorflow as tf
import yaml
import argparse
from read_params import read_param
from get_data import get_data
import numpy as np
import time
import model.train as training

def train_model(config_path):
    config = read_param(config_path)
    TRAIN_SIZE = config['data_load']['TRAIN_SIZE']
    BATCH_SIZE = config['data_load']['BATCH_SIZE']
    LEVEL_ITERS = config['train']['LEVEL_ITERS']

    LEVELS = [5, 4, 3, 2, 1, 0]


    for LEVEL in LEVELS:
        start_level = time.time()
        DSLR_SCALE = float(1)/(2**(LEVEL - 1))
        train_data, train_target = get_data(config_path, TRAIN_SIZE, DSLR_SCALE, True, False)
        print("TEST and TRAIN data loaded")

        print("TRAINING NETWORK")
        print("----------------------------------------------------")
        print("TRAINING LEVEL: ", LEVEL)
        print("----------------------------------------------------")

        iterations = LEVEL_ITERS[LEVEL]

        for i in range(iterations+1):
            start_iter = time.time()

            idx_train = np.random.randint(0, TRAIN_SIZE, BATCH_SIZE)
            phone_images = train_data[idx_train]
            dslr_images = train_target[idx_train]

            # random flips and rotations
            for k in range(BATCH_SIZE):
                random_rotate = np.random.randint(1, 100) % 4
                phone_images[k] = np.rot90(phone_images[k], random_rotate)
                dslr_images[k] = np.rot90(dslr_images[k], random_rotate)
                
                random_flip = np.random.randint(1, 100) % 2

                if random_flip == 1:
                    phone_images[k] = np.flipud(phone_images[k])
                    dslr_images[k] = np.flipud(dslr_images[k])
            
            training.fit(zip(phone_images, dslr_images), LEVEL, i)
            
            print('Time taken for epoch {} is {} sec\n'.format(i+1, time.time()-start))

            if i%1000 == 0:
                del train_data
                del train_target
                train_data, train_target = get_data(config_path, TRAIN_SIZE, DSLR_SCALE, True, False)

        print('Time taken for level {} is {} sec\n'.format(level, time.time()-start_level))

    print("TRAINING COMPLETE")
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml', help='params file')
    parsed_args = args.parse_args()
    train_model(config_path = parsed_args.config)