import tensorflow as tf
import numpy as np
import argparse


def build_model(input, instance_norm=True, instance_norm_level_1=False):
    # with tf.variable_scope("generator"):

        # Downsampling layers
        conv_l1_d1 = _conv_multi_block(
            input, 3, num_maps=32, instance_norm=False)
        pool1 = tf.nn.max_pool(conv_l1_d1, ksize=[1, 2, 2, 1], strides=[
                               1, 2, 2, 1], padding='VALID')

        conv_l2_d1 = _conv_multi_block(
            pool1, 3, num_maps=64, instance_norm=instance_norm)
        pool2 = tf.nn.max_pool(conv_l2_d1, ksize=[1, 2, 2, 1], strides=[
                               1, 2, 2, 1], padding='VALID')

        conv_l3_d1 = _conv_multi_block(
            pool2, 3, num_maps=128, instance_norm=instance_norm)
        pool3 = tf.nn.max_pool(conv_l3_d1, ksize=[1, 2, 2, 1], strides=[
                               1, 2, 2, 1], padding='VALID')

        conv_l4_d1 = _conv_multi_block(
            pool3, 3, num_maps=256, instance_norm=instance_norm)
        pool4 = tf.nn.max_pool(conv_l4_d1, ksize=[1, 2, 2, 1], strides=[
                               1, 2, 2, 1], padding='VALID')

        ############################
        # processing: level 5, input size: 14x14

        conv_l5_d1 = _conv_multi_block(
            pool4, 3, num_maps=512, instance_norm=instance_norm)
        conv_l5_d2 = _conv_multi_block(
            conv_l5_d1, 3, num_maps=512, instance_norm=instance_norm) + conv_l5_d1
        conv_l5_d3 = _conv_multi_block(
            conv_l5_d2, 3, num_maps=512, instance_norm=instance_norm) + conv_l5_d2
        conv_l5_d4 = _conv_multi_block(
            conv_l5_d3, 3, num_maps=512, instance_norm=instance_norm)

        conv_t4a = _conv_transpose_layer(conv_l5_d4, 256, 3, 2)  # 14 to 28
        conv_t4b = _conv_transpose_layer(conv_l5_d4, 256, 3, 2)  # 14 to 28

        # ouput: level 5
        conv_l5_out = _conv_layer(
            conv_l5_d4, 3, 3, 1, relu=False, instance_norm=False)
        output_l5 = tf.nn.tanh(conv_l5_out) * 0.58 + 0.58

        ############################
        # processing: level 4, input size: 28x28

        conv_l4_d2 = stack(conv_l4_d1, conv_t4a)
        conv_l4_d3 = _conv_multi_block(
            conv_l4_d2, 3, num_maps=256, instance_norm=instance_norm)
        conv_l4_d4 = _conv_multi_block(
            conv_l4_d3, 3, num_maps=256, instance_norm=instance_norm) + conv_l4_d3
        conv_l4_d5 = _conv_multi_block(
            conv_l4_d4, 3, num_maps=256, instance_norm=instance_norm) + conv_l4_d4
        conv_l4_d6 = _conv_multi_block(
            conv_l4_d5, 3, num_maps=256, instance_norm=instance_norm)
        conv_l4_d7 = stack(conv_l4_d6, conv_t4b)

        conv_l4_d8 = _conv_multi_block(
            conv_l4_d7, 3, num_maps=256, instance_norm=instance_norm)

        conv_t3a = _conv_transpose_layer(conv_l4_d8, 128, 3, 2)  # 28 to 56
        conv_t3b = _conv_transpose_layer(conv_l4_d8, 128, 3, 2)  # 28 to 56

        # output: level 4
        conv_l4_out = _conv_layer(
            conv_l4_d8, 3, 3, 1, relu=False, instance_norm=False)
        output_l4 = tf.nn.tanh(conv_l4_out) * 0.58 + 0.58

        ############################
        # processing: level 3, input size: 56x56

        conv_l3_d2 = stack(conv_l3_d1, conv_t3a)
        conv_l3_d3 = _conv_multi_block(
            conv_l3_d2, 5, num_maps=128, instance_norm=instance_norm) + conv_l3_d2
        # conv_l3_d4 = stack(conv_l3_d3, conv_l3_d2)
        conv_l3_d4 = _conv_multi_block(
            conv_l3_d4, 5, num_maps=128, instance_norm=instance_norm) + conv_l3_d3
        # conv_l3_d6 = stack(conv_l3_d5, conv_l3_d4)
        conv_l3_d5 = _conv_multi_block(
            conv_l3_d6, 5, num_maps=128, instance_norm=instance_norm) + conv_l3_d4
        # conv_l3_d8 = stack(conv_l3_d7, conv_l3_d6)
        conv_l3_d6 = _conv_multi_block(
            conv_l3_d5, 5, num_maps=128, instance_norm=instance_norm)

        conv_l3_d7 = stack(stack(conv_l3_d6, conv_l3_d1), conv_t3b)

        conv_l3_d8 = _conv_multi_block(
            conv_l3_d7, 3, num_maps=128, instance_norm=instance_norm)

        conv_t2a = _conv_transpose_layer(conv_l3_d8, 64, 3, 2)  # 56 to 112
        conv_t2b = _conv_transpose_layer(conv_l3_d8, 64, 3, 2)  # 56 to 112

        # output: level 3
        conv_l3_out = _conv_layer(
            conv_l3_d8, 3, 3, 1, relu=False, instance_norm=False)
        output_l3 = tf.nn.tanh(conv_l3_out) * 0.58 + 0.58

        ############################
        # processing: level 2, input size: 112x112
        conv_l2_d2 = stack(conv_l2_d1, conv_t2a)
        conv_l2_d3 = _conv_multi_block(
            conv_l2_d2, 5, num_maps=64, instance_norm=instance_norm)
        conv_l2_d4 = stack(conv_l2_d3, conv_l2_d1)

        conv_l2_d5 = _conv_multi_block(
            conv_l2_d4, 7, num_maps=64, instance_norm=instance_norm) + conv_l2_d4
        conv_l2_d6 = _conv_multi_block(
            conv_l2_d5, 7, num_maps=64, instance_norm=instance_norm) + conv_l2_d5
        conv_l2_d7 = _conv_multi_block(
            conv_l2_d6, 7, num_maps=64, instance_norm=instance_norm) + conv_l2_d6
        conv_l2_d8 = _conv_multi_block(
            conv_l2_d7, 7, num_maps=64, instance_norm=instance_norm)

        conv_l2_d9 = stack(conv_l2_d8, conv_l2_d1)

        conv_l2_d10 = _conv_multi_block(
            conv_l2_d9, 5, num_maps=64, instance_norm=instance_norm)

        conv_l2_d11 = stack(conv_l2_d10, conv_t2b)

        conv_l2_d12 = _conv_multi_block(
            conv_l2_d11, 3, num_maps=64, instance_norm=instance_norm)

        conv_t1a = _conv_transpose_layer(conv_l2_d12, 32, 3, 2)  # 112 to 224
        conv_t1b = _conv_transpose_layer(conv_l2_d12, 32, 3, 2)  # 112 to 224

        # output: level 2
        conv_l2_out = _conv_layer(
            conv_l2_d12, 3, 3, 1, relu=False, instance_norm=False)
        output_l2 = tf.nn.tanh(conv_l2_out) * 0.58 + 0.58

        ############################
        # processing: level 1, input size: 224x224

        conv_l1_d2 = stack(conv_l1_d1, conv_t1a)
        conv_l1_d3 = _conv_multi_block(
            conv_l1_d2, 5, num_maps=32, instance_norm=False)
        conv_l1_d4 = stack(conv_l1_d3, conv_l1_d1)
        conv_l1_d5 = _conv_multi_block(
            conv_l1_d4, 7, num_maps=32, instance_norm=False)

        conv_l1_d6 = _conv_multi_block(
            conv_l1_d5, 9, num_maps=32, instance_norm=instance_norm_level_1)
        conv_l1_d7 = _conv_multi_block(
            conv_l1_d6, 9, num_maps=32, instance_norm=instance_norm_level_1) + conv_l1_d6
        conv_l1_d8 = _conv_multi_block(
            conv_l1_d7, 9, num_maps=32, instance_norm=instance_norm_level_1) + conv_l1_d7
        conv_l1_d9 = _conv_multi_block(
            conv_l1_d8, 9, num_maps=32, instance_norm=instance_norm_level_1) + conv_l1_d8

        conv_l1_d10 = _conv_multi_block(
            conv_l1_d9, 7, num_maps=32, instance_norm=False)

        conv_l1_d11 = stack(conv_l1_d10, conv_l1_d1)

        conv_l1_d12 = _conv_multi_block(
            conv_l1_d11, 5, num_maps=32, instance_norm=False)

        conv_l1_d13 = stack(stack(conv_l1_d12, conv_t1b), conv_l1_d1)

        conv_l1_d14 = _conv_multi_block(
            conv_l1_d13, 3, num_maps=32, instance_norm=False)

        # output: level 1
        conv_l1_out = _conv_layer(
            conv_l1_d14, 3, 3, 1, relu=False, instance_norm=False)
        output_l1 = tf.nn.tanh(conv_l1_out) * 0.58 + 0.58

        ############################
        # Processing: level 0 (2x Upscaling), input size: 224x224

        conv_l0 = _conv_transpose_layer(conv_l1_d14, 8, 3, 2)   # 224 to 448
        conv_l0_out = _conv_layer(
            conv_l0, 3, 3, 1, relu=False, instance_norm=False)

        output_l0 = tf.nn.tanh(conv_l0_out) * 0.58 + 0.5

        return output_l0, output_l1, output_l2, output_l3, output_l4, output_l5
        # return output_l5


def stack(x, y):
    return tf.concat([x, y], 3)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# def leaky_relu(x, alpha=0.2):
#     return tf.maximum(x, alpha*x)


def _conv_init_vars(network, out_channels, filter_size, transpose=False):
    print(network.get_shape())
    # _, rows, cols, in_channels = [i.value for i in network.get_shape()]
    _, rows, cols, in_channels = [i for i in network.get_shape()]

    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.random.truncated_normal(
        weights_shape, stddev=0.01, seed=1), dtype=tf.float32)
    return weights_init


def _instance_norm(network):
    # batch, rows, cols, channels = [i.value for i in network.get_shape()]
    batch, rows, cols, channels = [i for i in network.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(network, [1, 2], keepdims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (network-mu)/(sigma_sq + epsilon)**(.5)

    return scale*normalized + shift


def _conv_layer(network, num_filters, filter_size, strides, relu=True, instance_norm=False, padding='SAME'):
    weights_init = _conv_init_vars(network, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))

    network = tf.nn.conv2d(network, weights_init,
                           strides_shape, padding=padding) + bias

    if instance_norm:
        network = _instance_norm(network)

    if relu:
        # network = leaky_relu(network)
        network = tf.keras.layers.LeakyReLU(alpha=0.2)(network)

    return network


def _conv_transpose_layer(network, num_filters, filter_size, strides):
    weights_init = _conv_init_vars(
        network, num_filters, num_filters, transpose=True)

    # batch_size, rows, cols, in_channels = [
    #     i.values for i in network.get_shape()]
    batch_size, rows, cols, in_channels = [
        i for i in network.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)

    strides_shape = [1, strides, strides, 1]

    network = tf.nn.conv2d_transpose(
        network, weights_init, tf_shape, strides_shape, padding='SAME')
    network = tf.keras.layers.LeakyReLU(alpha=0.2)(network)

    return network


def _conv_multi_block(input, max_size, num_maps, instance_norm):
    conv_3a = _conv_layer(input, num_maps, 3, 1, relu=True,
                          instance_norm=instance_norm)
    conv_3b = _conv_layer(conv_3a, num_maps, 3, 1,
                          relu=True, instance_norm=instance_norm)

    output_tensor = conv_3b

    if max_size >= 5:
        conv_5a = _conv_layer(input, num_maps, 5, 1,
                              relu=True, instance_norm=instance_norm)
        conv_5b = _conv_layer(conv_5a, num_maps, 5, 1,
                              relu=True, instance_norm=instance_norm)

        output_tensor = stack(output_tensor, conv_5b)

    if max_size >= 7:
        conv_7a = _conv_layer(input, num_maps, 7, 1,
                              relu=True, instance_norm=instance_norm)
        conv_7b = _conv_layer(conv_5a, num_maps, 7, 1,
                              relu=True, instance_norm=instance_norm)

        output_tensor = stack(output_tensor, conv_7b)

    if max_size >= 9:
        conv_9a = _conv_layer(input, num_maps, 9, 1,
                              relu=True, instance_norm=instance_norm)
        conv_9b = _conv_layer(conv_5a, num_maps, 9, 1,
                              relu=True, instance_norm=instance_norm)

        output_tensor = stack(output_tensor, conv_9b)

    return output_tensor


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml', help='params file')
    parsed_args = args.parse_args()

    phone_ = tf.keras.layers.Input(shape=(224, 224, 4), batch_size=1, dtype=tf.float32)
    print(phone_.shape)
    output_l0, output_l1, output_l2, output_l3, output_l4, output_l5 = build_model(phone_, instance_norm=True, instance_norm_level_1=False)
    # output_l5 = build_model(phone_, instance_norm=True, instance_norm_level_1=False)
    model = tf.keras.Model(inputs=phone_, outputs=output_l0, name='Assignee')
    # model = tf.keras.Model(inputs=phone_, outputs=output_l5, name='Assignee')
    print("Successfully built model")