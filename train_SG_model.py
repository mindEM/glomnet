import os
import h5py
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU, Dense, BatchNormalization, Dropout, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, concatenate, GlobalAveragePooling2D, Add, ZeroPadding2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import tensorflow.keras.backend as K
from tensorflow.keras.applications import Xception

class H5Datagenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 path_h5,
                 indexes,
                 batch_size = 1,
                 input_size = (1024, 1024, 3),
                 shuffle = True):
        self.path_h5 = path_h5
        self.indexes = indexes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(indexes)
        with h5py.File(self.path_h5, 'r') as f:
            self.num_classes = len(np.unique(f['labels']))
        
    def __getitem__(self, index):
        new_index = self.indexes[index]
        with h5py.File(self.path_h5, 'r') as f:
            orig = f['images'][new_index]
            msk = f['gt_heatmaps'][new_index]
            lab = f['labels'][new_index]
            
        lab = np.expand_dims(lab, axis = 0)
        lab = tf.keras.utils.to_categorical(lab, self.num_classes)
        return (np.expand_dims(orig, axis = 0), 
                {"seg_output": np.expand_dims(msk, axis = 0),
                 "clf_output1": lab,
                 "clf_output2": lab})
    
    def __len__(self):
        return self.n // self.batch_size
    
    def on_epoch_end(self):
        pass


# Please refer to README file for instructions on how to prepare dataset and cross-validation splits.
path_h5 = './dataset.h5'
with h5py.File(path_h5, 'r') as f:
    ncls = len(np.unique(f['labels']))
    
path_dset_splits = './cv_fold_0_dset_splits.json'
with open(path_dset_splits, 'r') as f:
    d = json.load(f)
    
train_indexes = d['train_indexes']
valid_indexes = d['valid_indexes']

traingen = H5Datagenerator(path_h5, train_indexes)
validgen = H5Datagenerator(path_h5, valid_indexes)


opt = 'SGD'
seg_w = .5000
clf_w1 = .0001
clf_w2 = .4999

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

# Pathes for logging and etc.
outfname = 'some_meaningful_name'
logfolder = './Graph/' + outfname

if not os.path.exists(logfolder):
    os.mkdir(logfolder)
    print("Logfolder created")
else:
    print("Logfolder found")
    
backbone = Xception(input_shape = (1024, 1024, 3),
                    weights = 'imagenet',
                    include_top = False)

i = backbone.input
start_neurons = 16
do = 0.2

conv4 = backbone.layers[121].output
o1 = GlobalAveragePooling2D()(conv4)
o1 = Dense(128, activation = 'relu')(o1)
o2 = Dropout(0.5)(o1)
clf_output1 = Dense(ncls,
                    activation = 'softmax',
                    name = 'clf_output1')(o2)


conv4 = LeakyReLU(alpha = 0.1)(conv4)
pool4 = MaxPooling2D((2, 2))(conv4)
pool4 = Dropout(do)(pool4)

# Middle
convm = Conv2D(start_neurons * 32, (3, 3),
               activation = None,
               padding = "same")(pool4)
convm = residual_block(convm, start_neurons * 32)
convm = residual_block(convm, start_neurons * 32)
convm = LeakyReLU(alpha = 0.1)(convm)

deconv4 = Conv2DTranspose(start_neurons * 16, (3, 3),
                          strides = (2, 2),
                          padding = "same")(convm)
uconv4 = concatenate([deconv4, conv4])
uconv4 = Dropout(do)(uconv4)

uconv4 = Conv2D(start_neurons * 16, (3, 3),
                activation = None,
                padding = "same")(uconv4)
uconv4 = residual_block(uconv4, start_neurons * 16)
uconv4 = residual_block(uconv4, start_neurons * 16)
uconv4 = LeakyReLU(alpha = 0.1)(uconv4)

deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3),
                          strides = (2, 2),
                          padding = "same")(uconv4)
conv3 = backbone.layers[31].output
uconv3 = concatenate([deconv3, conv3])    
uconv3 = Dropout(do)(uconv3)

uconv3 = Conv2D(start_neurons * 8, (3, 3),
                activation = None,
                padding = "same")(uconv3)
uconv3 = residual_block(uconv3, start_neurons * 8)
uconv3 = residual_block(uconv3, start_neurons * 8)
uconv3 = LeakyReLU(alpha = 0.1)(uconv3)

deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3),
                          strides = (2, 2),
                          padding = "same")(uconv3)
conv2 = backbone.layers[21].output
conv2 = ZeroPadding2D(((1, 0), (1, 0)))(conv2)
uconv2 = concatenate([deconv2, conv2])

uconv2 = Dropout(0.1)(uconv2)
uconv2 = Conv2D(start_neurons * 4, (3, 3),
                activation = None,
                padding = "same")(uconv2)
uconv2 = residual_block(uconv2, start_neurons * 4)
uconv2 = residual_block(uconv2, start_neurons * 4)
uconv2 = LeakyReLU(alpha = 0.1)(uconv2)

deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3),
                          strides = (2, 2),
                          padding = "same")(uconv2)
conv1 = backbone.layers[11].output
conv1 = ZeroPadding2D(((3, 0), (3, 0)))(conv1)
uconv1 = concatenate([deconv1, conv1])

uconv1 = Dropout(0.1)(uconv1)
uconv1 = Conv2D(start_neurons * 2, (3, 3),
                activation = None,
                padding = "same")(uconv1)
uconv1 = residual_block(uconv1, start_neurons * 2)
uconv1 = residual_block(uconv1, start_neurons * 2)
uconv1 = LeakyReLU(alpha = 0.1)(uconv1)

uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3),
                         strides = (2, 2),
                         padding = "same")(uconv1)   
uconv0 = Dropout(do)(uconv0)
uconv0 = Conv2D(start_neurons * 1, (3, 3),
                activation = None,
                padding = "same")(uconv0)
uconv0 = residual_block(uconv0, start_neurons * 1)
uconv0 = residual_block(uconv0, start_neurons * 1)
uconv0 = LeakyReLU(alpha = 0.1)(uconv0)

uconv0 = Dropout(do/2)(uconv0)
output_layer0 = Conv2D(start_neurons * 16, (1, 1),
                       padding = "same",
                       activation = "sigmoid",
                       name = 'export_grads')(uconv0)
seg_output = Conv2D(1, (1, 1), 
                    padding = "same",
                    activation = "sigmoid",
                    name = 'seg_output')(output_layer0)
upool1 = GlobalAveragePooling2D()(output_layer0)
udense1 = Dense(1024, activation = 'relu')(upool1)
udense1 = Dropout(0.5)(udense1)
clf_concat = concatenate([o1, udense1])
clf_output2 = Dense(ncls,
                    activation = 'softmax',
                    name = 'clf_output2')(clf_concat)

model = Model(i, [seg_output,
                  clf_output1,
                  clf_output2])


# Freeze pre-trained Xception backbone. This is not mandatory.
for n, layer in enumerate(backbone.layers):
    layer.trainable = False

losses = {
    "seg_output": "binary_crossentropy",
    "clf_output1": "categorical_crossentropy",
    "clf_output2": "categorical_crossentropy",
}

loss_weights = {"seg_output": seg_w,
                "clf_output1": clf_w1,
                "clf_output2": clf_w2}

model.compile(optimizer = opt,
              loss = losses,
              loss_weights = loss_weights,
              metrics = ['accuracy'])

model.summary()

checkpointer = ModelCheckpoint(f'./models/{outfname}.h5',
                               verbose = 1,
                               save_best_only = True)

earlystopper = EarlyStopping(monitor = 'val_clf_output2_loss',
                             patience = 50,
                             verbose = 1)

tbCallBack = TensorBoard(log_dir = logfolder,
                         histogram_freq = 0,
                         write_graph = False,
                         write_images = False)

model.fit(traingen,
          validation_data = validgen,
          batch_size = 1,
          epochs = 1000,
          verbose = 1,
          callbacks = [tbCallBack,
                       earlystopper,
                       checkpointer])
