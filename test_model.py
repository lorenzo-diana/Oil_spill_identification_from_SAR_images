from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse
import time
import sys
import os

from util_func import get_confusion_matrix
from util_func import compute_metrics
from util_func import output_to_rgb


PATH_PLOT           = './plot_test/'                # folder where plots will be saved
OUTPUT_SAVE         = './model_output/'             # folder where model'outputs will be saved
INFERENCE_TIME_FILE = './sample_inference_time.txt' # file where the inference time of all samples will be saved

# must match the file generated in generate_file_list.sh
test_file        = './test_set.txt'
# must match TEST_TILE_INPUT_PATH in generate_oil_dataset.py
test_folder      = './dataset/test_tile/'
# must match TEST_TILE_LABEL_PATH in generate_oil_dataset.py
seg_image_folder = './dataset/test_label_tile/'

def create_folders():
    try:
        os.makedirs(PATH_PLOT, exist_ok=True)
        os.makedirs(OUTPUT_SAVE, exist_ok=True)
    except Exception as ex:
        print(ex)
        exit(-1)

def plot_multiclass(img, mask, out, orig, img_name, classes, show_plot=0):
    for label_ID in range(0, mask.shape[2]): # img and mask must be channel last
        rgb_img = output_to_rgb(np.expand_dims(out[..., label_ID], axis = -1), np.expand_dims(mask[..., label_ID], axis = -1))
        tp, fp, fn, tn = get_confusion_matrix(np.expand_dims(out[..., label_ID], axis = -1), np.expand_dims(mask[..., label_ID], axis = -1))
        
        if tp+fp+fn != 0:
            IoU = tp/(tp+fp+fn)
            '''
            print('\nLabel ID: ', classes[label_ID])
            print('tp, fp, fn, tn: ', tp, fp, fn, tn)
            print('IoU: ', IoU)
            '''
        else:
            #print('\nLabel -1 ID: ', classes[label_ID])
            IoU = -1
        
        acc = (tp+tn)/(tp+fp+fn+tn)
        
        fig=plt.figure(figsize=(orig.shape[0], orig.shape[1]))
        plt.title('Label = '+classes[label_ID]+' - Input ('+img_name+') - Ground Truth - Output - Orig_out - Diff (FN = Green, FP = Red)   ---   IoU: '+str(IoU)+', Acc: '+str(acc))
        columns = len(classes)
        rows = 1

        # add specific parameters in imageshow for each class
        fig.add_subplot(rows, columns, 1)
        plt.imshow(img, vmin=0, vmax=255)
        
        fig.add_subplot(rows, columns, 2)
        plt.imshow(mask[..., label_ID], cmap='gray', vmin=0, vmax=1)
        
        fig.add_subplot(rows, columns, 3)
        plt.imshow(out[..., label_ID], cmap='gray', vmin=0, vmax=1)
        
        fig.add_subplot(rows, columns, 4)
        plt.imshow(orig[..., label_ID], cmap='gray', vmin=0, vmax=1)
        
        fig.add_subplot(rows, columns, 5)
        plt.imshow(rgb_img)
        
        if show_plot == 1:
            plt.show()
        if show_plot == 2:
            fig.savefig(PATH_PLOT+img_name+'_'+classes[label_ID]+'_IoU_'+str(IoU)+'__Acc_'+str(acc)+'.png', format='png')
        plt.close('all')

def preprocess_inputs(img): # add pre-processing as needed
    #    img = img/255. # normalize input if network input is float [0.0, 1.0]
    img = np.expand_dims(img, -1)
    return img

def test_multiclass_model(model, file_name, classes, show_plot=0, save_output=False, max_img=1000000):
    files = open(file_name, 'r')
    tot = 0
    mean_time = 0
    time_list = []
    
    num_classes = len(classes)
    
    mIoU = np.zeros(num_classes)
    tot_iou = np.zeros(num_classes)
    mAcc = np.zeros(num_classes)
    TP = np.zeros(num_classes)
    FP = np.zeros(num_classes)
    FN = np.zeros(num_classes)
    TN = np.zeros(num_classes)
    TP2 = np.zeros(num_classes)
    FP2 = np.zeros(num_classes)
    FN2 = np.zeros(num_classes)
    TN2 = np.zeros(num_classes)
    
    for line in files:
        line = line.strip()
        try:
            img_plot = Image.open(test_folder + line)
            img = np.array(img_plot, dtype=np.uint8)[..., 0] # data is int [0, 255]
            mask = np.load(seg_image_folder + line.replace('jpg', 'png')+'.npy').astype(np.float32)
        except:
            print(test_folder + line)
            print(seg_image_folder + line.replace('jpg', 'png')+'.npy')
            print()
            continue
        img = preprocess_inputs(img)

        if model.get_layer(index=0).input_shape[0][1:] != img.shape:
            print('ERROR!\nShape mismatch.')
            print('NN-input shape is: '+str(model.get_layer(index=0).input_shape[0][0]))
            print('Input data shape is: '+str(img.shape))
            exit()
        # check that shape of input and label is square and channel last
        assert img.shape[0] == img.shape[1]
        assert mask.shape[0] == mask.shape[1]
        sample_size = img.shape[0]*img.shape[1]
        
        img_ready = np.expand_dims(img,0)
        
        start_time = time.time()
        model_res = model.predict(img_ready)
        end_time = time.time()
        
        time_list.append(end_time - start_time)
        mean_time += time_list[-1]
        
        orig = out = (np.array(model_res)[0]).astype(np.float32)

        if save_output == True:
            np.save(output_save+line, out)

        out_i = np.argmax(out, -1) # out must be channel last
        new_out = np.zeros(out.shape, dtype=out.dtype)
        for i in range(0, out.shape[0]):
            for j in range(0, out.shape[1]):
                new_out[i,j,out_i[i,j]] = 1
        out = new_out
        
        plot_multiclass(img=img_plot, mask=mask, out=out, orig=orig, img_name=line, classes=classes, show_plot=show_plot)

        for i in range(num_classes):
            I, A, D, tp, fp, fn, tn = compute_metrics(out[..., i], mask[..., i])
            if (I != -1):
                mIoU[i] += I
                tot_iou[i] += 1
                TP[i] += tp
                FP[i] += fp
                FN[i] += fn
                TN[i] += tn
            mAcc[i] += A
            TP2[i] += tp
            FP2[i] += fp
            FN2[i] += fn
            TN2[i] += tn
    
        tot +=1
        if tot == max_img:
            break
    
    files.close()

    time_list_file = open(INFERENCE_TIME_FILE, 'w')
    for i in range(0, len(time_list)):
        time_list_file.write(str(time_list[i])+'\n')
    time_list_file.close()
    
    print('\n')
    print('Tot images:  ', tot)
    print('Mean inference time (s): ', mean_time/tot)
    
    over_iou = 0
    for i in range(num_classes):
        print('\n')
        print('Mean Acc['+classes[i]+']:        '+str(mAcc[i]/tot))
        print('Tot IoU images['+classes[i]+']:  '+str(tot_iou[i]))
        if tot_iou[i] != 0:
            over_iou += mIoU[i]/tot_iou[i]
            print('Mean IoU['+classes[i]+']:        '+str(mIoU[i]/tot_iou[i]))
            print('   TP_oil['+classes[i]+']:       '+str(round(TP[i]/(tot_iou[i]*sample_size)*100, 2)))
            print('   FP_oil['+classes[i]+']:       '+str(round(FP[i]/(tot_iou[i]*sample_size)*100, 2)))
            print('   FN_oil['+classes[i]+']:       '+str(round(FN[i]/(tot_iou[i]*sample_size)*100, 2)))
            print('   TN_oil['+classes[i]+']:       '+str(round(TN[i]/(tot_iou[i]*sample_size)*100, 2)))
        print('TP_tot['+classes[i]+']:          '+str(round(TP2[i]/(tot*sample_size)*100, 2)))
        print('FP_tot['+classes[i]+']:          '+str(round(FP2[i]/(tot*sample_size)*100, 2)))
        print('FN_tot['+classes[i]+']:          '+str(round(FN2[i]/(tot*sample_size)*100, 2)))
        print('TN_tot['+classes[i]+']:          '+str(round(TN2[i]/(tot*sample_size)*100, 2)))
    over_iou = over_iou/num_classes
    print('\n\nover_iou:  '+str(over_iou))

def check_positive(value):
    val = int(value)
    if val < 1:
        raise argparse.ArgumentTypeError(f"{val} is < 1")
    return val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', action='store', nargs=1, required=True, help='String - path to the .h5 model')
    parser.add_argument('-n', '--num_images', action='store', nargs=1, required=True, type=check_positive ,help="Integer. Max number of images to test")
    parser.add_argument('-p', '--plot', action='store', nargs=1, required=False, default=[0], type=int, choices=range(0, 3), help="Integer. Default 0. Generate the plot with input and output of the model for each output class. 0 do nothing, 1 to show the plots, or 2 to save the plots.")
    parser.add_argument('-s', '--save_output', action='store_true', required=False, default=False, help="Bool. If specified, the network output is saved as numpy file.")
    args = parser.parse_args()

    create_folders()
    
    model = load_model(args.model_name[0], compile=False)
    model.load_weights(args.model_name[0])
    model.summary()

    print('\n\nModel loaded.')
    print('Model name:          ', args.model_name[0])
    print('Max images to test:  ', args.num_images[0])
    print('Save plot:           ', args.plot[0])
    print('Save network output: ', args.save_output)
    print('\nStart testing...\n')
    
    classes = ['Sea', 'Oil', 'Look-alike', 'Ship', 'Land']
    test_multiclass_model(model=model, file_name=test_file, classes=classes, show_plot=args.plot[0], save_output=args.save_output, max_img=args.num_images[0])
