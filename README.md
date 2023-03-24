### Author: Lennart Seeger
### This file is licensed under the Apache 2.0 license available at
### http://www.apache.org/licenses/LICENSE-2.0

# IFEfNNS
This repository belongs to the master thesis of Lennart Seeger.
Different approaches for image feature extraction for a nearest neighbor search are implemented and evaluated. The overall structure of the approaches can be seen in the figure: ![Overall Structure](structure.jfif) For more insights into the code structure and information about the models, please have a look at the master thesis in this repository.

## Prerequisites
### Installing FAISS
To use the provided code, please first install FAISS under '../src/faiss/build/faiss/python/'. Otherwise, problems in the notebook "10_test_model.ipynb" can occur.

### Adding Classification Datasets
To use the notebooks appropriately with the evaluation datasets, please provide the mlrsnet, ucmerced, aid and custom denmark dataset to '../src/data/classification_datasets/[name_dataset]'

### Changing path to .tif file
To extract new data, please provide a GeoTIFF and change the referring paths (variable cog) in notebooks 02, 03 and 04.

# Code
## Notebooks
### Notebook 01
Notebook one is a notebook to process the classification datasets and the custom self-created dataset and to save it as a numpy archive for further computations by the other notebooks.

### Notebook 02
Notebook two is a notebook that extracts the indices of valid patches in a satellite image.

### Notebook 03
In Notebook 03, the in notebook 02 created indices are filtered to keep indices where the referring patches have a specific standard deviation.

### Notebook 04
The fourth notebook can extract patches from a 'tif' file using an index list created in notebooks 03 and 04.

### Notebook 05
The fifth notebook merges the datasets created in Notebook 04 into a single array. This helps computation in later steps.

### Notebook 06
In notebook sixth, the transfer Learning approach is used to train a model for later feature extraction.

### Notebook 07
The SIMCLR contrastive learning approach is implemented in the seventh notebook and outputs a model.

### Notebook 08
The NNCLR contrastive learning approach is implemented in the eighth notebook and outputs a model.

### Notebook 09
The MAE self-supervised learning approach is implemented in the ninth notebook and outputs a model.

### Notebook 10
Notebook three is the evaluation notebook consisting of a model to load initially and several evaluation parts.

## Assisting source code
Assisting Methods are in the src directory, split up into four groups.

### data
This contains everything necessary to handle and manage the data.

### model_utility
Different assisting functions, like the learning rate scheduler and Lars optimizer, can be found in the model utility.

### models
In this part, the main models are defined (SimCLR, NNCLR, MAE)

### supportive
Here, the supportive functionality is defined. This is mainly the coder necessary for the evaluation and visualization.