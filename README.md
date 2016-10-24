# SDM

Supervised Descent Method for Face Alignment using Python

First, download the dataset used by this project: 
link: https://pan.baidu.com/s/1jIJNg2q password: f36i

Second, get the data from data.tar that just downloaded

Third, put main.py in the same directory with the data folder, the run the main.py

For the first time, the main.py will run the train with a parameters, and after training process, you will get a train_data.mat file in the current directory. If you run the main.py with a train_data.mat file already there, the main.py will load the R,B,I from the file without the training process.

After you have get the R,B,I, you simply run the function test_after_run_main(n) to test the number nth image in the testset.
