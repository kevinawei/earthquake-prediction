# earthquake-prediction

Code + Submission for Kaggle Earthquake Prediction contest: https://www.kaggle.com/c/LANL-Earthquake-Prediction
Utilized a RNN model with 150 time steps using GRU rather than LSTM and with relu activation function.

The model has 3 layers, 1 CuDNNGRU layer and 2 dense layers that condense the dimension of the output to 10 and then 1(time to failure). 
The model was trained with adam optimizer and mae loss function to achieve the best possible results. 
It was trained for 30 epochs with 1000 steps per epoch and no dropout, as there was no overfitting in any of the tests.
