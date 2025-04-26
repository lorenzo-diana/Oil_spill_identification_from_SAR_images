import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.python.client import device_lib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import custom_generator as CG
import oil_spill_model as my_model


BATCH     = 4
BATCH_VAL = 16
lrate     = 0.002
EPOCHS    = 5
PATIENCE  = 10

# this folder will contain the history of the trained model
HISTORY_FOLDER      = './history_trained/'
# this folder will contain the plot of the history
PLOT_HISTORY_FOLDER = './plot_history/'
# this folder will contain the trained model
NN_WEIGHTS          = './nn_weights/'
# this folder will contain the checkpoints
NN_CHECKPOINTS      = './nn_checkpoints/'

def create_folders():
    try:
        os.makedirs(HISTORY_FOLDER, exist_ok=True)
        os.makedirs(PLOT_HISTORY_FOLDER, exist_ok=True)
        os.makedirs(NN_WEIGHTS, exist_ok=True)
        os.makedirs(NN_CHECKPOINTS, exist_ok=True)
    except Exception as ex:
        print(ex)
        exit(-1)

def l_rate_decay(curr_epoch):
    return float(lrate*tf.math.pow((1-(curr_epoch/EPOCHS)), 0.9))

def trainSegMulticlass(model, train_file, validation_file, model_name):
    list_metrics = ['categorical_accuracy', CG.keras_MIoU]
    loss_func = CG.categorical_focal_loss
    
    opt_func = Adam
    opt = opt_func(learning_rate = lrate, beta_1=0.9, beta_2=0.999)
    model.compile(loss = loss_func, optimizer = opt, metrics = list_metrics)

    x_train = CG.custom_image_generator_seg_multi_GPU(input_file=train_file, batch_size=BATCH, shuffle_epoch=False, shuffle_batch=True) # shuffle_epoch can be set to False when fit(shuffle=True)
    x_val = CG.custom_image_generator_seg_multi_GPU(input_file=validation_file, batch_size=BATCH_VAL, shuffle_epoch=False, shuffle_batch=True)  # shuffle_epoch can be set to False when fit(shuffle=True)

    checkpoint = ModelCheckpoint(filepath = NN_CHECKPOINTS+'weight_seg_'+model_name+".keras", verbose = 1, save_best_only = True, monitor='val_loss', mode='min')
    eStop = EarlyStopping(patience = PATIENCE, verbose = 1, restore_best_weights = True, monitor='val_loss')
    
    l_rate_schedule = LearningRateScheduler(l_rate_decay)

    history = model.fit(x_train, steps_per_epoch=len(x_train), epochs = EPOCHS, validation_data = x_val, validation_steps = len(x_val), shuffle=True, callbacks = [checkpoint, eStop, l_rate_schedule])

    return history

def plot_history(h, net_name, save_plot=False):
    list_keys = [s for s in h.keys() if 'val_' not in s and 'learning_rate' != s]
    print('Ready to', 'plot' if save_plot==False else 'save', ':', list_keys)
    for key in list_keys:
        plt.figure(key)
        plt.plot(h[key], 'o-')
        plt.plot(h['val_'+key], '^-')
        plt.title('model '+key)
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        if save_plot == True:
            plt.savefig(PLOT_HISTORY_FOLDER+net_name+'__'+key+'.png', format='png')
        else:
            plt.show()
    if save_plot == True:
        print('Plots saved in: '+PLOT_HISTORY_FOLDER)

if __name__ == '__main__':
    create_folders()

    model, net_name = my_model.oil_spill_net_3()
    model.summary()
    
    if tf.config.list_physical_devices('GPU'):
        print('Num GPUs available: ', len(tf.config.experimental.list_physical_devices('GPU')))
    else:
        print('GPU is not available!')
    print('Available devices:')
    print(device_lib.list_local_devices())
    print('\nTensorFlow version: ', tf.__version__)

    # must match the file generated in generate_file_list.sh
    path_input_images_train = './train_set.txt'
    # must match the file generated in generate_file_list.sh
    path_output_images_validation = './validation_set.txt'
    
    print('Start training...')
    # name of the trained model
    namesession = time.strftime("%Y-%m-%d_%H-%M")+'_'+net_name
    h = trainSegMulticlass(model, path_input_images_train, path_output_images_validation, namesession)
    model.save(NN_WEIGHTS+namesession+'.keras')
    np.save(HISTORY_FOLDER+'h_'+namesession,h.history)
    print('Net saved as: '+namesession+'.keras')
    plot_history(h.history, net_name, save_plot=True)
