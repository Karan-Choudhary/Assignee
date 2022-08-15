import argparse
import numpy as np
import yaml
import vgg
from utils import utils

def read_params(config_path):
    with open(config_path, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config


# MSE LOSS
def mse(enhanced_flat, dslr_flat, batch_size, TARGET_SIZE):
    loss_mse = tf.reduce_sum(tf.pow(dslr_flat - enhanced_flat, 2))/(TARGET_SIZE * batch_size)
    return loss_mse

# PSNR LOSS
def psnr(loss_mse):
    loss_psnr = 20 * utils.log10(1.0 / tf.sqrt(loss_mse))
    return loss_psnr

# SSIM LOSS
def ssim(enhanced, dslr_):
    loss_ssim = tf.reduce_mean(tf.image.ssim(enhanced, dslr_, 1.0))
    return loss_ssim

# MS-SSIM LOSS
def ms_ssim(enhanced, dslr_):
    loss_ms_ssim = tf.reduce_mean(tf.image.ssim_multiscale(enhanced, dslr_, 1.0))
    return loss_ms_ssim

# CONTENT LOSS
def content(vgg_dir, batch_size):
    CONTENT_LAYER = 'relu5_4'
    
    enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced * 255))
    dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(dslr_ * 255))

    content_size = utils._tensor_size(dslr_vgg[CONTENT_LAYER]) * batch_size
    loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER] - dslr_vgg[CONTENT_LAYER]) / content_size

    return loss_content

def loss(phone_, dslr_, enhanced, batch_size, LEVEL):
    config = read_params('params.yaml')

    PATCH_HEIGHT = config['data_load']['PATCH_HEIGHT']
    PATCH_WIDTH = config['data_load']['PATCH_WIDTH']

    vgg_dir = config['vgg_dir']

    DSLR_SCALE = float(1)/(2**(LEVEL - 1))
    TARGET_WIDTH = int(PATCH_WIDTH * DSLR_SCALE)
    TARGET_HEIGHT = int(PATCH_HEIGHT * DSLR_SCALE)
    TARGET_DEPTH = 3
    TARGET_SIZE = TARGET_HEIGHT * TARGET_WIDTH * TARGET_DEPTH


    enhanced_flat = tf.reshape(enhanced, [-1, TARGET_SIZE])
    dslr_flat = tf.reshape(dslr_, [-1, TARGET_SIZE])

    loss_mse = mse(enhanced_flat, dslr_flat, batch_size, TARGET_SIZE)
    loss_psnr = psnr(loss_mse)
    loss_ssim = ssim(enhanced, dslr_)
    loss_ms_ssim = ms_ssim(enhanced, dslr_)
    loss_content = content(vgg_dir, batch_size)

    if LEVEL == 5 or LEVEL == 4:
        loss_generator = loss_mse * 100
    if LEVEL == 3 or LEVEL == 2:
        loss_generator = loss_mse * 100 + loss_content
    if LEVEL == 1:
        loss_generator = loss_mse * 50 + loss_content
    if LEVEL == 0:
        loss_generator = loss_mse * 20 + loss_content+(1-loss_ssim) * 20
    
    return loss_generator, loss_ms_ssim