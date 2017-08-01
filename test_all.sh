#!/usr/bin/env bash


#just a bash script that will run everything on Dimitri's machine
#first lets clean any old processes that are lingering
./clean.sh

#run preprocessing
python main_thread.py -p -i /media/dperrin/pilot_images/ -s /media/dperrin/preprocessed/preprocessedTrain/ -l ./ -m ./  --model /media/dperrin/preprocessed/preprocessedTrain/model_data/ --pca 20 --sub 1


#run training
sudo python main_thread.py -t -i /media/dperrin/preprocessed/preprocessedTrain/ -s /media/dperrin/preprocessed/preprocessedTrain/ -l ./ -m ./  --model /media/dperrin/preprocessed/preprocessedTrain/model_data/ --sub 1 -w 0:1,1:20 -k 3


#run validation
sudo python main_thread.py -p -v -i /media/dperrin/val_images/ -s /media/dperrin/preprocessed/preprocessedVal/ -l ./ -m ./  --model /media/dperrin/preprocessed/preprocessedTrain/model_data/ --pca 20 --sub 1
