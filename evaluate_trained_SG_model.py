import h5py
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Conv2D, Add

def convolution_block(x, filters, size, 
                      strides = (1, 1),
                      padding = 'same',
                      activation = True):
    x = Conv2D(filters, size,
               strides = strides,
               padding = padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha = 0.1)(x)
        
    return x

def residual_block(blockInput, num_filters = 16):
    x = LeakyReLU(alpha = 0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation = False)
    x = Add()([x, blockInput])
    
    return x

# Load test set indexes.
outfname = 'your_saved_experiment_name'
path_json = f'./{outfname}_dset_splits.json'
with open(path_json, 'r') as f:
    d = json.load(f)
    
test_indexes = d['test_indexes']

# Load saved model.
model = load_model(f'./models/{outfname}.h5',
                   custom_objects = {'residual_block' : residual_block,
                                     'convolution_block' : convolution_block})

# Load & test images one-by-one.
hmaps, clf_aux, clf_main = [], [], []
with h5py.File('./dataset.h5', 'r') as f:
    for i in test_indexes:
        img = f['images'][i]
        r1, r2, r3 = model.predict(np.expand_dims(img, axis = 0))
        hmaps.append(r1)
        clf_aux.append(r2)
        clf_main.append(r3)
        

# Alternatively, if your hardware allows, load & test all images in a single batch:
'''with h5py.File('./dataset.h5', 'r') as f:
    test_images = f['images'][sorted(test_indexes)] # test_indexes must be sorted ascending!

hmaps, clf_aux, clf_main = model.predict(test_images)
'''

# Save results.
with h5py.File(f'./{outfname}_results.h5', 'w') as f:
    f.create_dataset('predicted_heatmaps', data = np.array(hmaps).squeeze())
    f.create_dataset('predicted_labels_aux', data = np.array(clf_aux).squeeze())
    f.create_dataset('predicted_labels_main', data = np.array(clf_main).squeeze())
