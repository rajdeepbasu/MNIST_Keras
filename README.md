# MNIST Digit Classification using Keras

Using keras to develop CNN to classify mnist handwritten digits

Convolutional Neural Network Architecture used :
[[*Convolution*] - > [*Convolution*] - > [*Batch* *Normalization*] - > [*Max* *Pooling*] - > [*Dropout*]] x 2 - > [[*Flatten*] - > [*Dense*] - > [*Dense*] - > [*Dropout*] - > [*Dense_Output*]]

This model will get you an accuracy of 99.29 on the kaggle competition which would get you a rank within the top 25% of the leaderboard.

The layers I used and Why I used them
1. **Convolution** - I used 64 'filters' each of size (3,3) to find out the features embedded in the images
2. **Batch** **Normalization** - I used this layer to speed up the overall training process since we can use a higher learning rate without the fear of losing accuracy. This layer can be used in place of dropout but can safely be avoided since we are dealing with only numbers
3. **Max** **Pooling** - I used this layer to '_downsample_' the image, reducing dimensionality and helps to identify features in the sub regions
4. **Dropout** - This layer basically acts as a regularization technique to reduce _overfitting_ and prevents complex relationship between neurons.
5. **Dense** - This layer acts as a hidden layer with a number of nodes to understand the relationship between the target variable and the image matrix generated.

The activation functions used and Why
1. **ReLu** - Also known as Rectified Linear Unit. The function mathematically can be represented as *max(0,x)*. This function basically adds non linearity to the linear relationships the neuron develop. ReLu deals with the problem of the *Vanishing Gradient* rather elegantly. In traditional gradient based methods, the back propagation would update the weights in the neural networks but what if the changes are minuscule then the weights wont change effectively and hence render stagnated learning. So to deal with this, ReLu was introduced.
![Image taken from https://towardsdatascience.com/activation-functions-in-neural-networks-58115cda9c96](https://cdn-images-1.medium.com/max/1400/1*JtJaS_wPTCshSvAFlCu_Wg.png)

2. **SoftMax** - The softmax function basically works for multi class classification. Softmax is essentially a sigmoid function extended to more than 2 classes. It can be represented as
![Image taken from Quora](https://qph.ec.quoracdn.net/main-qimg-5c7cbb4b9fa300ac1de0f1dc3568fa3c)

**Optimizer** - I used the **ADAM** or the *Adaptive Moment Estimation* optimizer instead of **Stochastic Gradient Descent**
In layman's terms, ADAM basically stores something called momentum as well. So for convex optimization algorithms, Stochastic Gradient Descent updates parameters randomly and finds out the gradient by calculating the second order differential which if negative moves to the minima. Adam basically add a decaying momentum value as well to speed up optimization process.
