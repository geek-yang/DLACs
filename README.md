[![DOI](https://zenodo.org/badge/181502808.svg)](https://zenodo.org/badge/latestdoi/181502808)
# DLACs :crystal_ball:
**Deep Learning Architecture for Climate science**, in short as **DLACs**, is a python library designed to implement deep learning algorisms to climate data for weather and climate prediction. Deep learning techniques to deal with spatial-temporal sequences, namely the Convolutional Long Short Term Memory neural netwroks (**ConvLSTM**), are implemented in this package. A probabilistic version of the structure is also employed, with an easy shift from ConvLSTM to Bayesian ConvLSTM (**BayesConvLSTM**) through Bayes by Backprop or Bernoulli approximation. <br/> 

Two types BayesConvLSTM are addressed here, which are BayesConvLSTM with variational inference (Blundell et. al. 2015; Shridhar et. al. 2019) and BayesConvLSTM by Bernoulli approximation with dropout (Gal and Ghahramani 2016).<br>

The module is designed to perform convolutional and recurrent operatiaons on structured climate data. It is built on pytorch.<br/>

## Function :computer:
Two kinds of deep neural networks structures are included by the package:<br>
* Convolutional Long Short Term Memory neural netwroks <br/>
* Bayesian Convolutional Long Short Term Memory neural netwroks <br/>

Two types of BayesConvLSTM are implemented here: BayesConvLSTM with variational inference and BayesConvLSTM approximated by Bernoulli distribution. The major differences are their functionality and the ways of training. BayesConvLSTM with variational inference is train by Bayes by Backprop (Blundell et. al. 2015; Shridhar et. al. 2019). BayesConvLSTM approximated by Bernoulli distribution is trained directly by back-propagation.<br/>

## Structure :file_folder:
Folder structure of the repositary:<br>
* `dlacs` Main components of DLACs
* `examples` Including python scripts showing the whole workflow of training and forecasting with BayesConvLSTM in DLACs
* `init` Sample ConvLSTM neural network used to initialize BayesConvLSTM
* `models` Examples of trained BayesConvLSTM neural networks
* `tests` Unit testing, functional testing and integration testing shown in jupyter notebooks
* `data` Forecast data with BayesLSTM and Variational Auto-Regressive Model

## Modules :floppy_disk:
Directory structure:<br>
* `ConvLSTM` Contains ConvLSTM layer and the forward module.
* `BayesConvLSTM` Contains BayesConvLSTM layer and the forward module of BayesConvLSTM with variational inference.
* `BayesConvLSTMBinary` Contains BayesConvLSTM layer and the forward module of BayesConvLSTM approximated by Bernoulli distribution.
* `function` Includes loss functions and distributions used by BayesConvLSTM.
* `metric` Scoring systems (RMSE, CRPS, etc.) to evaluate the performance of training.
* `preprocess` Functions to preprocess the input fields, like filtering, normalization, etc.
* `regrid` Moduels for geoscience / climate data regridding.
* `saveNetCDF` Modules to save the output in netCDF format.
* `visual` Visualization components for postprocessing and plotting.

## Dependency :books:
DLACs is tested on python 3.6 and has the following dependencies:<br>
* numpy
* matplotlib
* netCDF4
* scipy
* iris
* cartopy
* torch

## Configuration
Clone the repositary and add the repositary folder to your python system path, or just go to the repositary folder and run the following command after downloading:<br>
`python setup.py install` <br/>
For testing, please run:<br>
`python setup.py develop`<br/>
Note that this repository is still under construction. In case you find any bug or want to contact the author, please raise an issue and leave your comments.

## Cite our work :gift_heart:
Liu, Y. (2021). Arctic weather and climate: from mechanisms to forecasts. Wageningen University. https://doi.org/10.18174/545045
