#!/usr/bin/env python2
from __future__ import division, print_function

import sys
import caffe

def extract(model_fn):
    net = caffe.Net(model_fn, caffe.TEST)

if __name__ == '__main__':
    extract(sys.argv[1])
