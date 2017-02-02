#variables that can be used for setting paths, etc.
SAVE_DIR="preprocessedData/images/"
LOG_DIR="/preprocessedData/log/"
INPUT_DIR="/trainingData/"
METADATA_DIR="/metadata/"

echo "Creating directory for preprocessed files"
mkdir -p $SAVE_DIR

echo "Creating directory for log file"
mkdir -p $LOG_DIR


echo "Running Preprocessing script"
sudo python main_thread.py -p -i $INPUT_DIR -s $SAVE_DIR -l $LOG_DIR -m $METADATA_DIR


echo "DONE"
echo "Preprocessing Run Successfully. You may now DANCE :)"
