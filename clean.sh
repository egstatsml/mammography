#!/usr/bin/env bash


#script for cleaning up everything after I am done running my tests
#find the PIDS of the processes
TEMP=`pgrep -f main_thread.py`
kill -9 $TEMP

#remove all of the stored figures
rm -f ./figs/*.png
