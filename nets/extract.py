#!/usr/bin/env python2
from __future__ import division, print_function

import sys
import caffe
import json


def _blob_and_weights(net, layer_name):
    # Get the activation blob for this layer and its parameters
    # (weights).
    blob = net.blobs[net.top_names[layer_name][0]]
    weights = net.params[layer_name][0]
    return blob, weights

def extract(model_fn):
    # Load the model from the prototxt file.
    net = caffe.Net(model_fn, caffe.TEST)

    for name, layer in zip(net._layer_names, net.layers):
        layer_info = {
            'name': name,
            'type': layer.type,
        }

        # Convolutional layers.
        if layer.type in ('Convolution', 'Deconvolution'):
            blob, weights = _blob_and_weights(net, name)

            # Extract relevant hyperparameters from the layer's
            # activation and weight buffers.
            layer_height = blob.shape[2]
            layer_width = blob.shape[3]
            in_chan = weights.shape[0]
            out_chan = weights.shape[1]
            kernel_height = weights.shape[2]
            kernel_width = weights.shape[3]

            # Compute the total number of multiply-and-accumulate
            # operations for this convolutional layer.
            num_outputs = layer_width * layer_height * out_chan
            num_macs_per_out = in_chan * kernel_height * kernel_width
            num_macs = num_outputs * num_macs_per_out

            layer_info['macs'] = num_macs

        # Fully-connected layers.
        elif layer.type == "InnerProduct":
            blob, weights = _blob_and_weights(net, name)

            # Not sure about this at all.
            num_output = weights.shape[0]
            num_input = weights.shape[1]

            layer_info['macs'] = num_output * num_input

        yield layer_info

if __name__ == '__main__':
    out = list(extract(sys.argv[1]))
    print(json.dumps(out, indent=2, sort_keys=True))