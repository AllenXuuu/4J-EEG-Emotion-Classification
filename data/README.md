# Data 

A subset of [SJTU Emotion EEG Dataset (SEED)](https://bcmi.sjtu.edu.cn/home/seed/index.html) authored by BCMI lab led by **Prof.Bao-Liang Lu**.

Data can be downloaded from [this link](https://bcmi.cloud:5001/sharing/ArYaiQ2K9). Then put ```EEG_X.mat``` and ```EEG_Y.mat``` into ```./data``` folder.

It contains 15 human subjects in 3394 time steps. 310-dimensional differential entropy EEG-feature is collected for each human at each time step. Each 310-dimensional EEG-feature is annotated with a emotion label, including 3 categories (0 for clam, -1 for sad, 1 for happy).
