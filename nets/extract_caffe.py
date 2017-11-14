#!/usr/bin/env python3
import google.protobuf.text_format
import caffe_pb2
import sys
import json


def convert_layer(layer):
    return {
        "name": layer.name,
    }


def convert(infile):
    text = infile.read()
    net = google.protobuf.text_format.Merge(text, caffe_pb2.NetParameter())
    return {
        "name": net.name,
        "layers": [ convert_layer(l) for l in net.layers ]
    }


if __name__ == '__main__':
    out = convert(sys.stdin)
    print(json.dumps(out, indent=2, sort_keys=True))
