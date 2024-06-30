# Model Card and Project Description
We optimize the hyperparameters for a LeNet5-design based Convolutional Neural Network to predict the next-day share price.
We test the model with the share price time series of the Sylicon Valley Bank for the period before and after bankruptcy.

## Model Description
A LeNet5-design based Convolutional Neural Network.

**Input:**
The share prices' day Low, High, Close, Open, Adjusted Close time series are encoded into 32x32 images using [pyts summation Gramian angular field (GAF)](https://pyts.readthedocs.io/en/stable/auto_examples/image/plot_single_gaf.html) to obtain a temporal correlation between each pair of prices in the series.
Render of a GAF 32-day share price time series window for each feature.

We generate a stack of 32x32 images with shape (5, 491, 32, 32) which represents each of the 5 share price features' time series.
Each image represents a time series window of 32 days. We slide each window by 1 day from Ti to T(i+32) hence obtaining 491 time series windows or GAF images for each feature. These images are the inputs to the model.

The image dataset is split 80/20 into training/testing datasets. The actual share price for each window is its the next day share price.
The CNN is trained in mini-batches of 10 windows for each of the 5 features.

**Output:**
The next day predicted price.

**Model Architecture:**
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

## Performance

The score or performance of the model is measured for each

    accuracy = (correct price compared at 2 d.p / total) * 100

Through bayesian optimization we can achieve score 100% indicating overfitting.

## Limitations

The training of the model with similar prices (open, close, etc) may overtrain the model. igher generalization for the model may be achieved with a more focus implementation of regularization techniques and reducing the number of share price time series features and including still related idiosynchratic features such as volume traded and enterprise value. A larger share in the time series for a less volatile period with a lower trending nature can also be more representative of the central region of the distribution.

Further experimientation with additional hyperparameters in bayesian optimization may help increasing the score whilst reducing the number of parameters.