import os
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def save_img_color(img, path='.', form='png'):
    fig, axs = plt.subplots(figsize = (img.shape[0], img.shape[1]), dpi = 1)
    fig.subplots_adjust(0,0,1,1)
    axs.set_axis_off()
    axs.imshow(img, vmin=0, vmax=255) # cmap='gray'
    fig.savefig(path, format=form)
    plt.close(fig)

def gen_img_320_overlap(in_folder, out_folder):
    files = os.listdir(in_folder)
    for im in files:
        img = np.array(Image.open(in_folder + im), dtype=np.uint8)
        
        index = 0
        for i in range(0, 620, 320):
            for j in range(0, 1240, 310):
                ix = img[i:i+320, j:j+320, :]
                save_img_color(ix, out_folder+im+'_'+str(index), 'jpg')
                index += 1

def gen_label_5D(in_folder, out_folder): # from color image to one-hot encoding matrix
    files = os.listdir(in_folder)
    for im in files:
        img = np.array(Image.open(in_folder+im), dtype=np.uint8)
        out_label = np.zeros((img.shape[0], img.shape[1], 5), dtype=np.uint8)
        
        '''
        black = [0,0,0]
        green = [0,153,0]
        red   = [255,0,0]
        cyano = [0,255,255]
        brown = [153,76,0]
        '''
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                if ((img[i,j,:] == [0,0,0]).all()): # black pixel (sea)
                    out_label[i,j,0] = 1
                if ((img[i,j,:] == [0,255,255]).all()): # cyano pixel (oil spill)
                    out_label[i,j,1] = 1
                if ((img[i,j,:] == [255,0,0]).all()): # red pixel (look-alike)
                    out_label[i,j,2] = 1
                if ((img[i,j,:] == [153,76,0]).all()): # brown pixel (ship)
                    out_label[i,j,3] = 1
                if ((img[i,j,:] == [0,153,0]).all()): # green pixel (land)
                    out_label[i,j,4] = 1
        
        np.save(out_folder+im, out_label)

def gen_label_5D_320_overlap(in_folder, out_folder):
    files = os.listdir(in_folder)
    for im in files:
        img = np.load(in_folder+im)
        
        index = 0
        for i in range(0, 620, 320):
            for j in range(0, 1240, 310):
                ix = img[i:i+320, j:j+320, :]
                np.save(out_folder+im.replace('.npy', '')+'_'+str(index), ix)
                index += 1

def gen_flip_img(in_folder, out_folder):
    files = os.listdir(in_folder)
    for im in files:
        img = np.array(Image.open(in_folder + im), dtype=np.uint8)
        img_h = np.zeros((img.shape), dtype=np.uint8)
        img_o = np.zeros((img.shape), dtype=np.uint8)
        img_ho = np.zeros((img.shape), dtype=np.uint8)
        for i in range(0, img.shape[-1]):
            img_h[..., i] = np.flip(img[..., i], 0)
            img_o[..., i] = np.flip(img[..., i], 1)
            img_ho[..., i] = np.flip(img[..., i])
        save_img_color(img, out_folder+im, form='jpg')
        save_img_color(img_h.astype(np.uint8), out_folder+im+'_h', form='jpg')
        save_img_color(img_o.astype(np.uint8), out_folder+im+'_o', form='jpg')
        save_img_color(img_ho.astype(np.uint8), out_folder+im+'_ho', form='jpg')

def gen_flip_label(in_folder, out_folder):
    files = os.listdir(in_folder)
    for im in files:
        img = np.load(in_folder + im)
        img_h = np.zeros(img.shape, dtype=np.uint8)
        img_o = np.zeros(img.shape, dtype=np.uint8)
        img_ho = np.zeros(img.shape, dtype=np.uint8)
        for i in range(0, img.shape[-1]):
            img_h[..., i] = np.flip(img[..., i], 0)
            img_o[..., i] = np.flip(img[..., i], 1)
            img_ho[..., i] = np.flip(img[..., i])
        np.save(out_folder+im, img)
        np.save(out_folder+im.replace('.npy', '_h'), img_h.astype(np.uint8))
        np.save(out_folder+im.replace('.npy', '_o'), img_o.astype(np.uint8))
        np.save(out_folder+im.replace('.npy', '_ho'), img_ho.astype(np.uint8))



if __name__ == "__main__":
    print('Start...')

    augment_test_data = False
    
    ORIGINAL_DATASET_FOLDER       = './dataset/original_data/'
    PREPROCESSED_DATASET_FOLDER   = './dataset/'

    if os.path.isdir(ORIGINAL_DATASET_FOLDER) == False:
        print("Dataset folder not found: "+ORIGINAL_DATASET_FOLDER)
        print("Please ensure that the original dataset is located in the correct folder as described in the README file.")
        exit(-1)
    
    # train folders
    TRAIN_ORIGINAL_LABEL_PATH     = ORIGINAL_DATASET_FOLDER + 'train/labels/'
    TRAIN_CONVERTED_LABEL_PATH    = PREPROCESSED_DATASET_FOLDER + 'train_label/'
    TRAIN_ORIGINAL_INPUT_PATH     = ORIGINAL_DATASET_FOLDER + 'train/images/'
    TRAIN_TILE_LABEL_PATH         = PREPROCESSED_DATASET_FOLDER + 'train_label_tile/'
    TRAIN_TILE_INPUT_PATH         = PREPROCESSED_DATASET_FOLDER + 'train_tile/'
    TRAIN_AUGMENTED_LABEL_PATH    = PREPROCESSED_DATASET_FOLDER + 'train_label_tile_aug/'
    TRAIN_AUGMENTED_INPUT_PATH    = PREPROCESSED_DATASET_FOLDER + 'train_tile_aug/'
    
    # test folders
    TEST_ORIGINAL_LABEL_PATH      = ORIGINAL_DATASET_FOLDER + 'test/labels/'
    TEST_CONVERTED_LABEL_PATH     = PREPROCESSED_DATASET_FOLDER + 'test_label/'
    TEST_ORIGINAL_INPUT_PATH      = ORIGINAL_DATASET_FOLDER + 'test/images/'
    TEST_TILE_LABEL_PATH          = PREPROCESSED_DATASET_FOLDER + 'test_label_tile/'
    TEST_TILE_INPUT_PATH          = PREPROCESSED_DATASET_FOLDER + 'test_tile/'
    if (augment_test_data == True):
        TEST_AUGMENTED_LABEL_PATH     = PREPROCESSED_DATASET_FOLDER + 'test_label_tile_aug/'
        TEST_AUGMENTED_INPUT_PATH     = PREPROCESSED_DATASET_FOLDER + 'test_tile_aug/'
    
    try:
        os.makedirs(TRAIN_CONVERTED_LABEL_PATH, exist_ok=True)
        os.makedirs(TRAIN_TILE_LABEL_PATH, exist_ok=True)
        os.makedirs(TRAIN_TILE_INPUT_PATH, exist_ok=True)
        os.makedirs(TRAIN_AUGMENTED_LABEL_PATH, exist_ok=True)
        os.makedirs(TRAIN_AUGMENTED_INPUT_PATH, exist_ok=True)

        os.makedirs(TEST_CONVERTED_LABEL_PATH, exist_ok=True)
        os.makedirs(TEST_TILE_LABEL_PATH, exist_ok=True)
        os.makedirs(TEST_TILE_INPUT_PATH, exist_ok=True)
        if (augment_test_data == True):
            os.makedirs(TEST_AUGMENTED_LABEL_PATH, exist_ok=True)
            os.makedirs(TEST_AUGMENTED_INPUT_PATH, exist_ok=True)
    except Exception as ex:
        print(ex)
        exit(-1)
    
    ### TRAIN DATA ###
    # prepare the labels file
    gen_label_5D(TRAIN_ORIGINAL_LABEL_PATH, TRAIN_CONVERTED_LABEL_PATH)
    print('Gen train label 5D done.')
    # split the original images and labels into 320x320 tiles
    gen_img_320_overlap(TRAIN_ORIGINAL_INPUT_PATH, TRAIN_TILE_INPUT_PATH)
    print('Gen train img 320 overlap done.')
    gen_label_5D_320_overlap(TRAIN_CONVERTED_LABEL_PATH, TRAIN_TILE_LABEL_PATH)
    print('Gen train label 320 overlap done.')
    # data augmentation is done during pre-processing
    gen_flip_img(TRAIN_TILE_INPUT_PATH, TRAIN_AUGMENTED_INPUT_PATH)
    print('Train img flip done.')
    gen_flip_label(TRAIN_TILE_LABEL_PATH, TRAIN_AUGMENTED_LABEL_PATH)
    print('Train label flip done.')
    
    ### TEST DATA ###
    # prepare the labels file
    gen_label_5D(TEST_ORIGINAL_LABEL_PATH, TEST_CONVERTED_LABEL_PATH)
    print('Gen test label 5D done.')
    # split the original images and labels into 320x320 tiles
    gen_img_320_overlap(TEST_ORIGINAL_INPUT_PATH, TEST_TILE_INPUT_PATH)
    print('Gen test img 320 overlap done.')
    gen_label_5D_320_overlap(TEST_CONVERTED_LABEL_PATH, TEST_TILE_LABEL_PATH)
    print('Gen test label 320 overlap done.')
    # data augmentation is done during pre-processing
    if (augment_test_data == True):
        gen_flip_img(TEST_TILE_INPUT_PATH, TEST_AUGMENTED_INPUT_PATH)
        print('Test img flip done.')
        gen_flip_label(TEST_TILE_LABEL_PATH, TEST_AUGMENTED_LABEL_PATH)
        print('Test label flip done.')
    
    try:
        shutil.rmtree(TRAIN_CONVERTED_LABEL_PATH, ignore_errors=True)
        shutil.rmtree(TRAIN_TILE_LABEL_PATH, ignore_errors=True)
        shutil.rmtree(TRAIN_TILE_INPUT_PATH, ignore_errors=True)

        shutil.rmtree(TEST_CONVERTED_LABEL_PATH, ignore_errors=True)
        if (augment_test_data == True):
            shutil.rmtree(TEST_TILE_LABEL_PATH, ignore_errors=True)
            shutil.rmtree(TEST_TILE_INPUT_PATH, ignore_errors=True)
    except Exception as ex:
        print(ex)
        exit(-1)
    
    print("Done.")
