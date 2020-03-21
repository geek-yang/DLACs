#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='dlacs',
    version='0.1.1',
    description="Deep Learning Architecture for Climate science, in short as DLACs, is a python library designed to implement deep learning algorisms to climate data for weather and climate prediction. Deep learning techniques to deal with spatial-temporal sequences, namely the Convolutional Long Short Term Memory neural netwroks (ConvLSTM), are implemented in this package. A probabilistic version of the structure is also employed, with an easy shift from ConvLSTM to Bayesian ConvLSTM (BayesConvLSTM).",
    long_description=readme + '\n\n',
    author="Yang Liu",
    author_email='y.liu@esciencecenter.nl',
    url='https://github.com/geek-yang/DLACs',
    packages=[
        'dlacs',
    ],
    package_dir={'dlacs':
                 'dlacs'},
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='dlacs',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
)
