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

We run up to 10,000 epochs and optimize the number of outputs for the Convolution Layer 1 and 2, learning rate and Dropout probability for 10 steps of bayesian optimization and steps of random exploration. See optimizer_results.txt for these results. The models are saved in /bayesian_optimization_saved_models:

    pbounds = {'output_conv_1': (40, 80), 'output_conv_2': (8, 16), 'learning_rate': (0.00001, 0.0001), 'dropout_probab': (0.0, 0.5), 'momentum': (0.8, 1.0)}

This is only an initial choice in the search space.

## RESULTS
The model predicts at low accuracy. Literature indicates a LetNet design is not optimal to fit the data.

Bayesian optimization results helped to manually explore higher accuracy hyper-parameter and model parameters.

Different model designs, in particular a Long short-term memory model (LTSM) may be more suited for this prediction task.

Best Bayesian optimization test dataset highest accuracy performance achieves a score of 3.125% for:

    'params': {'dropout_probab': 0.49443054445324736, 'learning_rate': 7.733490889418554e-05, 'momentum': 0.8560887984128811, 'output_conv_1': 71.57117313805955, 'output_conv_2': 8.825808052621136}

This result helps us manually explore hyper-parameters and model parameter optimal results, achieving 17.91% accurancy. The predicted-to-actual predict price difference to 2.dp is 95%. We acknowledge this may be an unnecessarily too high a threhold to determine this difference:

    'params': {'dropout_probab': 0, 'learning_rate': 0.0001, 'momentum': 0.9, 'output_conv_1': 40, 'output_conv_2': 12}

## ACKNOWLEDGEMENTS
I thank [Yahoo Finance](https://pypi.org/project/yfinance/) for the time series data provided. I also thank for the inspiration [repo](https://github.com/ShubhamG2311/Financial-Time-Series-Forecasting). I also thank the [BayesianOptimization library s_opt module](https://github.com/bayesian-optimization/BayesianOptimization).
