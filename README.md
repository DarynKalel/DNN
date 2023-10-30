# DNN for a classification problem, Overfitting and L1-L2 regularization

In this project, I explored how to develop a simple Deep Neural Network for a classification problem using the one of the most popular libraries in Python "PyTorch". 
Then, I explored how to face a well known problem that is common during the training phase, called overfitting on the training set and techniques to solve that issue.

# Dataset

In this project, I used the AG News Subset that is available in Torchtext Datasets. The AG's news topic classification dataset is constructed by choosing the 4 largest topic classes (1-World, 2-Sports, 3-Business, 4-Sci/Tech) from a larger news corpus. The total number of training samples is  120.000  and testing  7.600 . For resource constraints, I limited the number of training, validation and test samples to  10.000 ,  1.000  and  1.000 , respectively.

# Model Definition and Training

I defined the 3-layers feed forward model! In PyTorch we do that by subclassing nn.Module, and initialize the neural network layers in __init__. Every nn.Module subclass implements the operations on input data in the forward method.

In detail, I created a FeedforwardNetwork class. The constructor takes three arguments: input_dim (an integer), which is the dimension of the input layer; num_classes (an integer), which is the number of output classes; and hidden_layers_dim (a list of integers), which is the dimension of the hidden layers in the network.

In the constructor, the nn.ModuleList() object is initialized to hold the layers of the network. If hidden_layers_dim is an empty list, then the network only consists of a single linear layer (from input to output). Otherwise, the network consists of multiple linear layers. The first layer goes from the input layer to the first hidden layer, and the subsequent hidden layers go from the previous hidden layer to the next hidden layer. The final output layer goes from the last hidden layer to the output layer.

The _init_weights function is a helper function that initializes the weights of the linear layers in the network. It initializes the weights using a normal distribution with a mean of 0 and a standard deviation of 0.1, and sets the biases to zero. 

The forward function defines the forward pass of the network. It first checks whether the network has only one layer (in which case it simply returns the output of that layer). Otherwise, it applies the relu activation function to the output of each hidden layer and passes the result to the next layer. Finally, it returns the output of the final layer. 

# Overfiting

A common problem that occurs when you train a deep neural network is overfitting. Overfitting occurs when you achieve a good fit of your model on the training data, while it does not generalize well on new, unseen data. In other words, the model learned patterns specific to the training data, which are irrelevant in other data. For research constraints I tried to modify the training parameters in order to have a model that overfits.

# L1 - and  L2 -regularization

One possible way to solve the overitting issue is by using regularization methods. The two most common regularization methods in Deep Learning are the L1-norm regularization and the L2-norm regularization. I also applied an early stoppage, one of the forms of Regularization, which works in the following way: it monitors the generalization error of one model and stop training when generalization error begins to degrade

