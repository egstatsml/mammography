#!/usr/bin/python

import threading
import Queue
import time
from my_thread import *

#import my classes
from breast import breast
from feature_extract import feature
from read_files import spreadsheet

descriptor = spreadsheet(training=True, run_synapse = False)
thread_list = ["Thread-1", "Thread-2"]
name_list = ['one', 'two']
threads = []
queue_lock = threading.Lock()
work_queue = Queue.Queue(10)
id = 0

# Create new threads
for ii in thread_list:
    thread = my_thread(id, ii, 1)
    thread.start()
    threads.append(thread)
    id += 1

queue_lock.acquire()
for word in name_list:
        work_queue.put(word)
        queue_lock.release()
        



