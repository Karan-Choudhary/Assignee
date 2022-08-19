import tensorflow as tf
from model.model import build_model
from model.loss import loss
import yaml
import time

# Optimizer
optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

def read_param(config_path):
    with open(config_path, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

config = read_param('params.yaml')
# PATCH_HEIGHT = config['data_load']['PATCH_HEIGHT']
# PATCH_WIDTH = config['data_load']['PATCH_WIDTH']
BATCH_SIZE = config['data_load']['BATCH_SIZE']

# DSLR_SCALE = float(1)/(2**(LEVEL - 1))
# TARGET_WIDTH = int(PATCH_WIDTH * DSLR_SCALE)
# TARGET_HEIGHT = int(PATCH_HEIGHT * DSLR_SCALE)
# TARGET_DEPTH = 3
# TARGET_SIZE = TARGET_HEIGHT * TARGET_WIDTH * TARGET_DEPTH


# phone_ = tf.keras.Input(shape=(PATCH_HEIGHT, PATCH_WIDTH, 4), batch_size=BATCH_SIZE, dtype=tf.float32)
# dslr_ = tf.keras.Input(shape=(TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH), batch_size=BATCH_SIZE, dtype=tf.float32)

model_l0, model_l1, model_l2, model_l3, model_l4, model_l5 = build_model()
models = [model_l0, model_l1, model_l2, model_l3, model_l4, model_l5]

@tf.function()
def train(phone_, dslr_, LEVEL):

    with tf.GradientTape() as tape:

        if LEVEL == 5:
            model = model_l5
        if LEVEL == 4:
            model = model_l4
        if LEVEL == 3:
            model = model_l3
        if LEVEL == 2:
            model = model_l2
        if LEVEL == 1:
            model = model_l1
        if LEVEL == 0:
            model = model_l0
        
        enhanced = model(input_image)
        loss_generator = loss(phone_, dslr_, enhanced, BATCH_SIZE, LEVEL)
    
    gradients = tape.gradient(loss_generator, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def fit(train_ds, LEVEL):
    config = read_param('params.yaml')
    SAVE_MODEL = config['model_dir']

    LEVEL_ITER = config['train']['level_iter']
# fullmetal alchemist ritherhood

    LEVELS = [5, 4, 3, 2, 1, 0]
    for level in LEVELS:
        start_level = time.time()
        epochs = LEVEL_ITER[level]
        for epoch in range(epochs+1):
            start = time.time()

            for n, (input_image, target_image) in train_ds.enumerate():
                print('.',end="")
                if(n+1)%100==0:
                    print()
                train(input_image, target_image, level)
            print()

            if (epoch+1)%5 == 0:
                models[level].save(SAVE_MODEL[level] + '_' + str(epoch+1) + '.h5')
            
            print('Time taken for epoch {} is {} sec\n'.format(epoch+1, time.time()-start))
        
        print('Time taken for level {} is {} sec\n'.format(level, time.time()-start_level))

            
                