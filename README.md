# About

This repository contains code to train a Graph Neural Network to predict
whether a given bee in a hive survives a given number of days into the future
using age, location, and relational information about interactions between
pairs of bees. Used libraries include **PyTorch Geometric**, **PyTorch Lightning**,
and **Optuna**.

# How to run the scripts

## Locally

1. Either create the directories ```/kaggle/working/``` and ```/kaggle/input/honey-bee-temporal-social-network/```, or change ```INPUT_PATH``` and ```OUTPUT_PATH``` in ```./source/config.py```.
2. Download the files ```bee_daily_data.csv``` and ```interaction_networks_20160729to20160827.h5``` at the bottom of the page https://doi.org/10.5281/zenodo.4438013 under 'Files' and move them into the directory at ```INPUT_PATH```.
3. ```pip install optuna torch torch_geometric pytorch_lightning scikit-learn matplotlib h5py numpy pandas swifter```
4. In ```./source/config.py```, set the ```MODEL_NAME``` to 'MLP' for vanilla neural network, or 'GNN' for graph convolutional neural network. Other run parameters can also be changed there.
5. To skip hyperparameter tuning, comment out the first line in ```./source/best_hyperparameters```, and uncomment the ```best_hyperparameters``` dictionary in that file.
6. For all the imports to work properly, set an environment variable via ```export BEESDIR='/full/path/to/local/repository'```, with '/full/path/to/local/repository' replaced accordingly.
7. Run ```python ./source/experiment.py```

## On Kaggle

1. Create a Kaggle dataset called 'your-user-name/bees-source' containing all the scripts in ```./source/```.
2. Create a Kaggle script from  ```./source/experiments.py```, and add both your newly created dataset from step 1., and [the bees dataset](https://zenodo.org/record/4438013) as inputs.
3. Run the Kaggle script created in 2.

The MLP and GNN code were both tested to run with **Python 3.7.12** on GPU. In addition, the MLP code was tested to run with Python **3.9.12** on CPU.

# Data

Wild, Benjamin, Dormagen, David and Landgraf, Tim (2021) Social networks predict the life and death of honey bees -
Data [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4438013.svg)](https://doi.org/10.5281/zenodo.4438013)
