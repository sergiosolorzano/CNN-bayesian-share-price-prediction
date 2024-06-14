## Convolutional Neural Network with Bayesian hyperparameter optimization to predict next day share price from a stock price time series

## Description
We optimize the hyperparameters for a LeNet5-design based Convolutional Neural Network to predict the next-day share price.
We test the model with the share price time series of the Sylicon Valley Bank for the period before and after bankruptcy.
The repo runs on python and leverages available pytorch libraries.

The share prices' day Low, High, Close, Open, Adjusted Close time series are encoded into 32x32 images using [pyts summation Gramian angular field (GAF)](https://pyts.readthedocs.io/en/stable/auto_examples/image/plot_single_gaf.html) to obtain a temporal correlation between each pair of prices in the series.
Render of a GAF 32-day share price time series window for each feature:
<img width="1045" alt="image" src="https://github.com/sergiosolorzano/CNN-bayesian-share-price-prediction/assets/24430655/985af796-f2d1-43c2-98e9-86e9610262dc">

Render average of the above GAF images:

<img width="225" alt="image" src="https://github.com/sergiosolorzano/CNN-bayesian-share-price-prediction/assets/24430655/27cb4600-58c8-42ca-8968-d0a1b6d99586">

We generate a stack of 32x32 images with shape (5, 491, 32, 32) which represents each of the 5 share price features' time series.
Each image represents a time series window of 32 days. We slide each window by 1 day from Ti to T(i+32) hence obtaining 491 time series windows or GAF images for each feature.

The image dataset is split 80/20 into training/testing datasets. The actual share price for each window is its the next day share price.
The CNN is trained in mini-batches of 10 windows for each of the 5 features.

This repo is my choice for the end of course project at [Professional Certificate in Machine Learning and Artificial Intelligence](https://execed-online.imperial.ac.uk/professional-certificate-ml-ai)

## DATA
We use [Yahoo Finance](https://pypi.org/project/yfinance/) python package and historical daily share price database for the period 2021-10-01 to 2023-12-01.

## MODEL 
A LeNet5-design based Convolutional Neural Network which includes:
+ 1 Convolution Layer 1: It's output is processed through a Rectified Linear Unit ReLU activation function and Max Pool kernel.
+ 1 Convolution Layer 2: It's output is processed through a ReLU activation function and Max Pool kernel.
+ 1 Fully Connected Layer 1: It's output is processed through a ReLU activation function.
+ 1 Fully Connected Layer 2: It's output is processed through a ReLU activation function.
+ 1 Fully Connected Layer 3: It's output is processed through a ReLU activation function.
+ filter_size_1=(2, 2) applied to Convo 1
+ filter_size_2=(2, 3) applied to Max Pool
+ filter_size_3=(2, 3) applied to Convo 2
+ stride=2 for convo layers

The model incorporates drop out regularization on the fully connected layers.

The choice of model used leverages prior work and there is no other particular reason but to test the concept.

## HYPERPARAMETER OPTIMSATION
The repo optimizes the model's hyperparameters leveraging [BayesianOptimization library s_opt module](https://github.com/bayesian-optimization/BayesianOptimization).

We run 20 epochs and optimize the number of outputs for the Convolution Layer 1 and 2, learning rate and Dropout probability for 10 steps of bayesian optimization and steps of random exploration:

    pbounds = {'output_conv_1': (40, 80), 'output_conv_2': (8, 16), 'learning_rate': (0.001, 0.01), 'dropout_probab': (0.0, 0.5)}

This is only an initial choice in the search space.

## RESULTS
The model can overfit possibly due to the highly correlated price series features used to train. 

Higher generalization for the model may be achieved with a more focus implementation of regularization techniques and reducing the number of share price time series features and including still related idiosynchratic features such as volume traded and enterprise value. A larger share in the time series for a less volatile period with a lower trending nature can also be more representative of the central region of the distribution.

Best Bayesian optimization test dataset performance achieves a score of 100% for:

    'params': {'dropout_probab': 0.34325046384079183, 'learning_rate': 0.008511631047076357, 'output_conv_1': 40.73153109376767, 'output_conv_2': 14.00115451955974}}

## ACKNOWLEDGEMENTS
I thank [Yahoo Finance](https://pypi.org/project/yfinance/) for the time series data provided. I also thank for the inspiration [repo](https://github.com/ShubhamG2311/Financial-Time-Series-Forecasting). I also thank the [BayesianOptimization library s_opt module](https://github.com/bayesian-optimization/BayesianOptimization).
