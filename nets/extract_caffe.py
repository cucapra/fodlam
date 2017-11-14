#!/usr/bin/env python3
import google.protobuf.text_format
import google.protobuf.json_format
import caffe_pb2
import sys

def convert(infile):
    text = infile.read()
    net = google.protobuf.text_format.Merge(text, caffe_pb2.NetParameter())
    return google.protobuf.json_format.MessageToJson(net)

if __name__ == '__main__':
    print(convert(sys.stdin))
