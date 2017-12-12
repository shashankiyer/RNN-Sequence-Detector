# RNN-Sequence-Detector

RNN's can be used for sequence detection. Therefore, a network was developed to observe a stream of data points and classify the math function correctly as either 'Linear', 'Sinusoidal', 'Random' or 'Exponential'.

Instructions to follow while using the code:
Rnn_train.py accepts a set of parameters, namely
1.	The mode- train/test
2.	The number of training samples that the generator must create
3.	The learning rate
4.	The name of the Tensorflow model to use while testing or to create while training
5.	The regularization to apply to the weights
6.	The number of hidden units in the LSTM
A sample of the set of arguments is as follows 
python rnn_train.py train 5000 0.01 model cross 10
