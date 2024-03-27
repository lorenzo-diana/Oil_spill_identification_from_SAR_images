import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence

class custom_image_generator_seg_multi_GPU(Sequence):
    def __init__(self, input_file, bs):
        self.in_file = input_file
        self.batch_size = bs
    
    def __len__(self):
        return int(np.ceil(file_len(self.in_file) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        # must match the path of TRAIN_AUGMENTED_INPUT_PATH in generate_oil_dataset.py
        train_folder     = './dataset/train_tile_aug/'
        # must match the path of TRAIN_AUGMENTED_LABEL_PATH in generate_oil_dataset.py
        seg_image_folder = './dataset/train_label_tile_aug/'

        f = open(self.in_file, "r")
        for _ in range(idx*self.batch_size):
            line = f.readline()
        images = []
        labels = []
        while len(images) < self.batch_size:
            line = f.readline()
            if line == "":
                f.seek(0)
                line = f.readline()

            line = line.strip()
            try:
                image = np.array(Image.open(train_folder + line), dtype=np.uint8)[..., 0]
                image = np.expand_dims(image, -1) # 0 -> channel_first ; -1 -> channel_last
                label = np.load(seg_image_folder + line.replace('jpg', 'png') + '.npy')
                label = np.asarray(label, dtype=np.float32)
                images.append(image)
                labels.append(label)
            except Exception as ex:
                print('failed at idx: ', idx)
                print(ex)
                print('file image: ' + train_folder + line)
                print('file label: ' + seg_image_folder + line.replace('jpg', 'png') + '.npy')
                continue
        
        images = np.array(images)
        labels = np.array(labels)
        
        # return the batch to the calling function
        return (images, labels)


def file_len(name): # count only non-empty lines
    num_lines = 0
    with open(name) as f:
        for i, l in enumerate(f):
            if len(l) > 1:
                num_lines += 1
    return num_lines

keras_MIoU_var = tf.keras.metrics.MeanIoU(num_classes=5)
def keras_MIoU(y_true, y_pred):
    y_true = tf.math.argmax(y_true, axis=-1, output_type=tf.dtypes.int32)
    y_pred = tf.math.argmax(y_pred, axis=-1, output_type=tf.dtypes.int32)
    _ = keras_MIoU_var.update_state(y_true, y_pred)
    return keras_MIoU_var.result()

# This implementation of the focal loss was taken from:
# https://github.com/aldi-dimara/keras-focal-loss/blob/master/focal_loss.py
"""
Implementation of Focal Loss from the paper in multiclass classification
Formula:
    loss = -alpha*((1-p)^gamma)*log(p)
Parameters:
    alpha -- the same as wighting factor in balanced cross entropy
    gamma -- focusing parameter for modulating factor (1-p)
Default value:
    gamma -- 2.0 as mentioned in the paper
    alpha -- 0.25 as mentioned in the paper
"""
def categorical_focal_loss(y_true, y_pred):
    # compute softmax if not already done in the NN
    #e = K.exp(y_pred)
    #y_pred = e/K.sum(e, axis=-1, keepdims=True)
    alpha=0.75
    gamma=2.0
    # Define epsilon so that the backpropagation will not result in NaN
    # for 0 divisor case
    epsilon = K.epsilon()
    # Clip the prediction value
    y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
    # Calculate cross entropy
    cross_entropy = -y_true*K.log(y_pred)
    # Calculate weight that consists of  modulating factor and weighting factor
    weight = alpha*y_true*K.pow((1-y_pred), gamma)
    # Calculate focal loss
    loss = weight*cross_entropy
    # Sum the losses in mini_batch
    loss = K.sum(loss, axis=-1)
    return loss
