# Author
Xinyu XU   
518021910645
# Environment 
My experiment environment is
```
python 3.6
cuda 10.0
torch 1.1
numpy
scipy
```

# Code Structure 
```
-- data
    |--EEG_X.mat
    |--EEG_Y.mat
    |
-- weights
    |--run_IDN_best.pth (weights of pre-trained IDN)
    |--run_IDN_best.txt (result of IDN pre-train)
    |
    |--run_LSTM_best.pth (weights of whole IDN+LSTM model)
    |--run_LSTM_best.txt (result of classification)
    |
-- run_IDN.py
-- run_LSTM.py
-- test_model.py
-- run_svm.py
-- tools.py
-- datautils.py
-- models.py
```


# Test model
To test the model in my submission, use command
```
python test_model.py --IDN_LSTM_weight weights/run_LSTM_best.pth
```
It will restore the weight in ```weights/run_LSTM_best.pth``` and make evaluation.

# Reproduce training

If you want to reproduce the training.

First do the unsupervised pretraining of IDN by command
```
python run_IDN.py --lambda_rec 1e-3 --lambda_dom 1e-2 --lambda_cross 1e-2 --lambda_mmd 1 --epoch 100
```
Then the weight after first stage training will be stored in ```weights/run_IDN.pth```

Second, we are going to do classification with LSTM based on pre-trained weight. 

Take my submitted ```weights/run_IDN_best.pth``` for example. 

Use command
```
python run_LSTM.py --lambda_cls 0.01 --epoch 300 --IDN_weight weights/run_IDN_best.pth
```
Then the final weight will be stored in ```weights/run_LSTM.pth```

# Contact
Any problem about the code, feel free to e-mail ```xuxinyu2000@sjtu.edu.cn```

Alternatively, my wechat is ```18815201763```
