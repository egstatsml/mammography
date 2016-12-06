#!/usr/bin/python

import threading
import Queue
import time
from my_thread import my_thread

#import my classes
from breast import breast
from feature_extract import feature
from read_files import spreadsheet






descriptor = spreadsheet(training=True, run_synapse = False)
name_list = descriptor.filenames
threads = []
id = 0

# Create new threads
for ii in range(0,5):
    thread = my_thread(id)
    thread.start()
    threads.append(thread)
    id += 1
    
    
#setting up the queue for all of the threads, which contains the filenames
for filename in name_list:
    my_thread.q_lock.acquire()
    my_thread.q.put(filename)
    my_thread.q_lock.release()


#now some code to make sure it all runs until its done

#keep this main thread open until all is done
#just looking at the first (zeroth) thread, doesn't matter which one we look at
#they all share the same queue
while (not my_thread.q.empty()):
    pass

#we are done so set the exit flag to true
my_thread.exit_flag = True
#wait until all threads are done
for t in threads:
    t.join()
print "Exiting Main Thread"
