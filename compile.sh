#!/usr/bin/env bash

#am using this bash script in a makefile like context
TARGET=breast_cython

cython -a $TARGET.pyx

gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing       -I/usr/include/python2.7 -I ~/anaconda2/pkgs/numpy-1.11.1-py27_0/lib/python2.7/site-packages/numpy/core/include -o $TARGET.so $TARGET.c
