# Summary
This repository holds the Python code for a method proposed to determine types of kidney glomerular damage.
* The repository provides code for the following steps:
    * Generation of ground truth heatmaps for spatially guided attention training.
    * Dataset preprocessing.
    * Building, training, and using spatially-guided glomeruli classifier.

* This repository does not cover:
    * Glomeruli detection and cropping from a digital whole slide pathology image (partially done using HALO software [^halo]).
    * Manual steps such as glomeruli annotation/labeling.
    * Training of standard classifiers (e.g. original Xception model) and extraction of attention maps.   

[^halo]:[Halo by Indica labs](https://indicalab.com/halo/)

## Principle of spatially-guided glomeruli classifier
Build and train the U-Net-like architecture using pre-trained Xception as an encoder. U-Net with a pre-trained encoder is not a new concept - to mention a few published papers [^polyp_paper] [^road_paper] [^pneumothorax_paper] as well as a few repositories here on GitHub [^github_unet_zoo] [^github_pytroch_seg_mods].
[^polyp_paper]:[Mohammed et al. "Y-Net: A deep Convolutional Neural Network for Polyp Detection"](https://arxiv.org/pdf/1806.01907.pdf)
[^road_paper]:[Zhou et al. "D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction"](https://ieeexplore.ieee.org/document/8575492)
[^pneumothorax_paper]:[Abedalla et al. "Chest X-ray pneumothorax segmentation using U-Net with EfficientNet and ResNet architectures"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8279140/)
[^github_unet_zoo]:["U-Net Keras Zoo"](https://github.com/daniCh8/unet-keras-zoo)
[^github_pytroch_seg_mods]:["Segmentation models with pretrained backbones. PyTorch."](https://github.com/qubvel/segmentation_models.pytorch)

* Some custom features and modifications implemented here include:
    * U-net decoder branch is trained on pre-generated heatmaps based on simple annotations placed by a human expert.
    * U-net has two additional output layers that perform label classification.
    * In general, U-Net is set to output:
        * Single-layer heatmap bearing the object localization information (localization heatmap).
        * Auxiliary label prediction via a classification layer branching out at the end of the pre-trained encoder (prior knowledge).
        * Main label prediction via the second classification layer that aggregates (by global average pooling) feature maps from a final convolutional layer of the decoder. Thus, its output is guided by a localization heatmap (decoder output).
    * Both main and auxiliary classification channels are interconnected via a skip connection.
    * The total training loss is a weighted aggregation of losses from the three output channels - BCE for the U-Net decoder output, and categorical cross-entropy for both classification outputs.

## Heatmap generation & storage
Notebook `./generate_GT_heatmaps.ipynb` holds all functions and examples (including visuals) necessary to produce ground truth heatmaps and store the training dataset. The saved dataset of images, labels, and ground truth heatmaps is meant to be stored in hdf5 file[^hdf5] with the following structure:
[^hdf5]: [h5py user manual](https://docs.h5py.org/en/stable/)
```python
dataset.h5/
├── labels       # shape = (N, ),              type = ('ndarray', dtype('int64'))
├── images       # shape = (N, 1024, 1024, 3), type = ('ndarray', dtype('float64'))
└── gt_heatmaps  # shape = (N, 1024, 1024),    type = ('ndarray', dtype('float64'))

where N =  number of images.
```

Model training relies on a custom data generator that feeds the model with data from the dataset.h5 file.
In this example, images are of 1024 x 1024 pixels shape, and of 'float' data type (this is just for convenience - neither of these is mandatory). Notebook also holds different examples of freeform annotations applied by an expert.

## Dataset splitting for cross-validation
The dataset can be split into cross-validation folds. Image ids of each fold are stored inside JSON dictionary with the following structure:
```python
cv_fold_0_dset_splits.json/
├── train_indexes # type = 'list', values dtype('int64')
├── valid_indexes # type = 'list', values dtype('int64')
└── test_indexes  # type = 'list', values dtype('int64')
```
During training the 'traingen' and 'validgen' data generators are initiated with your `./dataset.h5` file and a list of corresponding indexes from JSON dictionary. Notebook `./cross_validation_dataset_splitting.ipynb` holds an example of dataset splitting and index storing.


# Imports/Dependencies:
```python
h5py == 3.5.0
jsonschema == 4.1.2
matplotlib == 3.4.3
numpy == 1.21.3
scikit-image == 0.19.3
scikit-learn == 0.24.2
tensorboard == 2.7.0
tensorflow == 2.7.0
```

# Usage
* Generate ground truth heatmaps (see `./generate_GT_heatmaps.ipynb` file for reference) and store the dataset.
* Build, train, and save the model (see `./train_SG_model.py` file for reference).
    * To monitor the training start tensorboard:
    `tensorboard --logdir ./Graph --host localhost --port 6006`
* Load trained model and process new glomeruli images (see `./evaluate_trained_SG_model.py` file for reference).
