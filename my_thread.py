#!/usr/bin/python

import threading
import time
import numpy as np

class my_thread (threading.Thread):
    
    #variable that is shared amongst all of the my_thread object
    self.counter = 2
    
    def __init__(self, thread_id, name, counter):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name
        self.counter = counter
        
    def run(self):
        print "Starting " + self.name
        # Get lock to synchronize threads
        self.process()
        self.increment()
    
    
    
    """
    increment()
    
    Description:
    Will put a lock on the thread and increment the file number counter
    Lock is used so we dont increment multiple times by accident
    
    """
    def increment(self):
        self.thread lock.acquire()
        self.counter += 1
        #Free lock to release next thread
        self.thread_lock.release()
        
        
    def process(self):
        #wait for a random amount of time
        time.sleep(np.random.randint(1,5))
        
        
    
    
    
    
    """
    print_time()
    
    Description:
    Will just print the time required for the thread to run
    
    """
    
    def print_time(threadName, delay, counter):
        while counter:
            time.sleep(delay)
            print "%s: %s" % (threadName, time.ctime(time.time()))
            counter -= 1

