#!/usr/bin/env bash

#variables that can be used for setting paths and creating directories

#input dir is the directory where the preprocessed scans are stored
PREPROCESSED_DIR="/preprocessedData/preprocessedTrain/"
SAVE_DIR_ROOT="/scratch"
SAVE_IMAGE_DIR="$SAVE_DIR_ROOT/preprocessedTrain/"
SAVE_DIR_MODEL_FILES="$SAVE_IMAGE_DIR/model_data/"
LOG_DIR="/modelState"
INPUT_DIR="/trainingData/"
METADATA_DIR="/metadata/"
MODEL_STATE_DIR="/modelState"

echo "TRAINING PART OF THE MODEL"
echo "Creating directory in scratch space where I can save things temporarily"
mkdir -p $SAVE_DIR_MODEL_FILES
echo "listing all of the home directories"
ls /


echo "Running Training Script"
python main_thread.py -t -i $PREPROCESSED_DIR -s $SAVE_IMAGE_DIR -l $LOG_DIR -m $METADATA_DIR --model $MODEL_STATE_DIR --sub 1 --pca 20 -w 0:1,1:2 -k 1 -c


#save the model in the modelstate dir
#echo "Copying the model to the modelState dir"
cp $SAVE_IMAGE_DIR/ $MODEL_STATE_DIR/ -r -f



echo "DONE"
echo "Training Run Successfully. You may now DANCE :)"
