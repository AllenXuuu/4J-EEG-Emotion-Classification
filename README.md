# 4J-EEG-Emotion-Classification

Course project for EI328 Science and Technology Innovation 4J (Parallel Machine Learning with Application to Large-Scale Data Mining), tutored by **Prof.Bao-Liang Lu**.

Propose a domain generalization solution via feature manipulation to personalize EEG-based emotion classification.

[[PPT](./slides.pptx)]  [[Paper](./paper.pdf)]


# Data 

A subset of [SJTU Emotion EEG Dataset (SEED)](https://bcmi.sjtu.edu.cn/home/seed/index.html) authored by BCMI lab led by **Prof.Bao-Liang Lu**.

Data can be downloaded from [this link](https://bcmi.cloud:5001/sharing/ArYaiQ2K9). Then put ```EEG_X.mat``` and ```EEG_Y.mat``` into ```./data``` folder.

It contains 15 human subjects in 3394 time steps. 310-dimensional differential entropy EEG-feature is collected for each human at each time step. Each 310-dimensional EEG-feature is annotated with a emotion label, including 3 categories (0 for clam, -1 for sad, 1 for happy).

# Train

First do the unsupervised pretraining of IDN by command
```
python run_IDN.py --lambda_rec 1e-3 --lambda_dom 1e-2 --lambda_cross 1e-2 --lambda_mmd 1 --epoch 100
```
Then the weight after first training stage will be stored in ```weights/run_IDN.pth```

Second, we are going to do classification with LSTM based on pre-trained weight. 
```
python run_LSTM.py --lambda_cls 0.01 --epoch 300 --IDN_weight weights/run_IDN.pth
```
Then the final weight will be stored in ```weights/run_LSTM.pth```


# Test
```
python test_model.py --IDN_LSTM_weight weights/run_LSTM.pth
```
It will restore the weight in ```weights/run_LSTM.pth``` and make evaluation.