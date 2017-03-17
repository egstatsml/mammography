#!/usr/bin/env bash

#variables that can be used for setting paths and creating directories
SAVE_DIR_ROOT="/preprocessedData"
SAVE_IMAGE_DIR="$SAVE_DIR_ROOT/preprocessedTrain/"
SAVE_DIR_MODEL_FILES="$SAVE_IMAGE_DIR/model_data/"
LOG_DIR="/modelState"
INPUT_DIR="/trainingData/"
METADATA_DIR="/metadata/"
MODEL_STATE_DIR="/modelState"

echo "TRAINING PART OF THE MODEL"
echo "Running Training Script"
python main_thread.py -t -f -i $SAVE_IMAGE_DIR -s $SAVE_IMAGE_DIR -l $LOG_DIR -m $METADATA_DIR -a $SAVE_DIR_MODEL_FILES


#save the model in the modelstate dir
#echo "Copying the model to the modelState dir"
cp $SAVE_DIR_MODEL_FILES/model_file/ $MODEL_STATE_DIR/model_file/ -r -f


#sudo python main_thread.py -t -i /media/dperrin/preprocessed/preprocessedTrain/ -s /media/dperrin/preprocessed/preprocessedTrain/ -l ./ -m ./  -a /media/dperrin/preprocessed/preprocessedTrain/model_data/ -f


echo "DONE"
echo "Training Run Successfully. You may now DANCE :)"
