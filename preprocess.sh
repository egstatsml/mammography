#!/usr/bin/env bash

#variables that can be used for setting paths and creating directories
SAVE_DIR_ROOT="/preprocessedData"
SAVE_IMAGE_DIR="$SAVE_DIR_ROOT/preprocessedTrain/"
SAVE_DIR_MODEL_FILES="$SAVE_IMAGE_DIR/model_data/"
LOG_DIR="/modelState"
INPUT_DIR="/trainingData/"
METADATA_DIR="/metadata/"
MODEL_STATE_DIR="/modelState"

echo "Creating directory where preprocessed images will be saved"
mkdir -p $SAVE_IMAGE_DIR

echo "Creating directory for features found during the preprocessing stage"
mkdir -p $SAVE_DIR_MODEL_FILES

echo "Creating directory for log file"
mkdir -p $LOG_DIR


echo "Running Preprocessing script"
python main_thread.py -p -i $INPUT_DIR -s $SAVE_IMAGE_DIR -l $LOG_DIR -m $METADATA_DIR -a $SAVE_DIR_MODEL_FILES


#save the model in the modelstate dir
#echo "Copying the model to the modelState dir"
#cp $SAVE_DIR_MODEL_FILES/model_file $MODEL_STATE_DIR/model_file

echo "DONE"
echo "Preprocessing Run Successfully. You may now DANCE :)"
