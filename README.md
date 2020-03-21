[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3722946.svg)](https://doi.org/10.5281/zenodo.3722946)
# DLACs :crystal_ball:
**Deep Learning Architecture for Climate science**, in short as **DLACs**, is a python library designed to implement deep learning algorisms to climate data for weather and climate prediction. Deep learning techniques to deal with spatial-temporal sequences, namely the Convolutional Long Short Term Memory neural netwroks (**ConvLSTM**), are implemented in this package. A probabilistic version of the structure is also employed, with an easy shift from ConvLSTM to Bayesian ConvLSTM (**BayesConvLSTM**). <br/> Two ways of realization of the Bayesian deep learning are addressed here, which are Bayes by Backprop (Blundell et. al. 2015; Shridhar et. al. 2019) and Bayesian deep learning with dropout (Gal and Ghahramani 2016).  

The module is designed to perform convolutional and recurrent operatiaons on structured climate data. It is built on pytorch.<br/>

## Function :computer:
Two kinds deep neural networks structures are included by the package:
* Convolutional Long Short Term Memory neural netwroks <br>
* Bayesian Convolutional Long Short Term Memory neural netwroks <br>

## Dependency :books:
DLACs is tested on python 3.6 and has the following dependencies:
* numpy
* matplotlib
* netCDF4
* scipy
* iris
* cartopy
* torch

## Modules :floppy_disk:
Directory structure:
* `ConvLSTM` Contains ConvLSTM layer and the forward module.
* `BayesConvLSTM` Contains BayesConvLSTM layer and the forward module.
* `example` Examples of how to implement the network structures in this library.
* `models` Examples of trained models.
* `tests` Includes unit testing, functional testing and integration testing.

## Cite our work :gift_heart:
DOI:10.5281/zenodo.3722946
