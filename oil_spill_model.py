from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Softmax

def oil_spill_net():
    print("\n\n\nDue to copyright issues, the actual CNN model has not been uploaded to this repository. For more information, please contact the corresponding author of the paper mentioned in the README file.\n\n\n")
    input("Press Enter to continue...")


    input_image = Input(shape=(320,320,1))
    x = Softmax()(input_image)
    output = x
    
    model = Model(input_image, output)
    return model, 'oil_spill_net'
