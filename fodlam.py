#!/usr/bin/env python3
from __future__ import division, print_function

import os
import csv
import json
import sys
from collections import namedtuple

# The networks that our accelerators have measurements for.
NETWORKS = ('VGG16', 'AlexNet')
LAYER_KINDS = ('conv', 'fc')

# Accelerator data files.
DATA_DIR = 'data'
EIE_FILE = 'eie-layers.csv'
EYERISS_FILES = {
    'VGG16': 'eyeriss-vgg16.csv',
    'AlexNet': 'eyeriss-alexnet.csv',
}

# EIE reports latencies in microseconds; Eyeriss in milliseconds. Eyeriss
# reports per-layer power in milliwatts.
EIE_TIME_UNIT = 10 ** (-6)
EYERISS_TIME_UNIT = 10 ** (-3)
EYERISS_POWER_UNIT = 10 ** (-3)

# Process nodes for published implementations. Both use TSMC processes.
EIE_PROCESS_NM = 45
EYERISS_PROCESS_NM = 65

# EIE reports only a total design power (in watts).
EIE_POWER = 0.59

# Data files with neural network statistics.
NETS_DIR = 'nets'
NET_FILES = {
    'VGG16': 'VGG_ILSVRC_16_layers_deploy.json',
    'AlexNet': 'alexnet_deploy.json',
}

# Caffe's names for layer kinds.
CAFFE_KINDS = {
    "InnerProduct": "fc",
    "Convolution": "conv",
    "Deconvolution": "conv",
}

# Two kinds of layer specs. Lookup layers are precise; we just need to
# look up their costs from the base data. Scale layers are
# approximations; we need to use the average cost per MAC.
LookupLayer = namedtuple('LookupLayer', ['net', 'layer'])
ScaleLayer = namedtuple('ScaleLayer', ['kind', 'macs'])


def load_hw_data():
    """Load the published numbers from our data files. Return a dict
    with base values reflecting EIE and Eyeriss layer costs.
    """
    # Load EIE data (latency only).
    eie_latencies = {}
    with open(os.path.join(DATA_DIR, EIE_FILE)) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Layer'] == 'Actual Time':
                for k, v in row.items():
                    # The table has the network and the layer name
                    # together in one cell.
                    if ' ' in k:
                        network, layer = k.split()
                        if network in NETWORKS:
                            eie_latencies[network, layer] = \
                                float(v) * EIE_TIME_UNIT

    # Load Eyeriss data (latency and energy).
    eyeriss = {
        'latency_total': {},
        'latency_proc': {},
        'power': {},
    }
    for network in NETWORKS:
        with open(os.path.join(DATA_DIR, EYERISS_FILES[network])) as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer = row['Layer']
                if layer == 'Total':
                    continue
                eyeriss['latency_total'][network, layer] = \
                    float(row['Total Latency (ms)']) * EYERISS_TIME_UNIT
                eyeriss['latency_proc'][network, layer] = \
                    float(row['Processing Latency (ms)']) * EYERISS_TIME_UNIT
                eyeriss['power'][network, layer] = \
                    float(row['Power (mW)']) * EYERISS_POWER_UNIT

    return { 'eie': eie_latencies, 'eyeriss': eyeriss }


def layer_costs(published):
    """Get the latencies (in seconds) and power (in watts) for *all*
    layers in VGG-16 by combining EIE and Eyeriss data.
    """
    eie_lat = published['eie']
    eyeriss_lat = published['eyeriss']['latency_total']
    eyeriss_pow = published['eyeriss']['power']

    # Process scaling factor between Eyeriss and EIE. We scale the EIE
    # numbers because the magnitudes for Eyeriss are more significant
    # and the paper has a more complete evaluation.
    proc_scale = EYERISS_PROCESS_NM / EIE_PROCESS_NM
    eie_lat_scaled = { k: v * proc_scale for k, v in eie_lat.items() }
    eie_power_scaled = EIE_POWER * (proc_scale ** 2)

    # Combine the latencies for all the layers.
    latency = dict(eie_lat_scaled)
    latency.update(eyeriss_lat)

    # For Eyeriss, we have per-layer power numbers. For EIE, from the paper:
    # "Energy is obtained by multiplying computation time and total measured
    # power". So we follow their lead and assume constant power.
    power = { k: eie_power_scaled for k in eie_lat }
    power.update(eyeriss_pow)

    return latency, power


def norm_layer_name(name):
    """Some heuristics to normalize a layer name from multiple sources.

    For example, some depictions of VGG-16 use use upper case; others
    use lower case. Some use hyphens; others use underscores. These
    heuristics are by no means complete, but they increase the
    likelihood that layer names from multiple sources will align.
    """
    return name.upper().replace('_', '-')


def load_net(filename):
    """Load layer statistics for a single network from a JSON file.
    Return a mapping from layer names to ScaleLayer tuples.
    """
    with open(os.path.join(NETS_DIR, filename)) as f:
        layers = json.load(f)

    # Flatten the list of layer statistics dictionaries into a
    # name-to-number mapping.
    out = {}
    for layer in layers:
        if 'macs' in layer:
            name = norm_layer_name(layer['name'])
            kind = CAFFE_KINDS[layer['type']]
            out[name] = ScaleLayer(kind, layer['macs'])
    return out


def load_net_data():
    """Load statistics about the neural networks from our description
    files. Return mappings from layer names to ScaleLayers.
    """
    return { network: load_net(filename)
             for network, filename in NET_FILES.items() }


def scaling_ratios(net_data, costs):
    """Get the scaling ratio---the cost per MAC---for convolutional and
    fully-connected layers with the given cost set.
    """
    # Total numerators and denominators.
    totals = {
        'conv': { 'cost': 0, 'macs': 0 },
        'fc': { 'cost': 0, 'macs': 0 },
    }

    # Sum up the cost and MAC counts for each layer type.
    for net, layer_stats in net_data.items():
        for layer, stats in layer_stats.items():
            cost = costs[net, layer]
            totals[stats.kind]['macs'] += stats.macs
            totals[stats.kind]['cost'] += cost

    # Return ratios.
    return { k: v['cost'] / v['macs'] for k, v in totals.items() }


def dict_product(a, b):
    """Pointwise-multiply the values in two dicts with identical sets of
    keys.
    """
    assert set(a.keys()) == set(b.keys())
    return { k: v * b[k] for k, v in a.items() }


def select_sum(keys, mapping):
    """Sum the values in `mapping` corresponding to keys that are
    present in `keys`. Every key in `keys` must be present in `mapping`.
    """
    assert set(keys) <= set(mapping)
    return sum(v for k, v in mapping.items() if k in keys)


def load_config(config_file):
    """Load a neural network configuration from a file-like object.
    Return a set of enabled layers, which are instances of either
    `LookupLayer` or `ScaleLayer`.
    """
    config = json.load(config_file)
    if "net" in config:
        # A "built-in" (precise) network.
        return [LookupLayer(config["net"], norm_layer_name(l))
                for l in config['layers']]
        return layers

    elif "netfile" in config:
        # A "new" (scaled) network. Load the statistics for this network
        # from its file.
        net_stats = load_net(config["netfile"])
        return [net_stats[l] for l in config['layers']]

    else:
        assert False


def load_params():
    """Load and set up all the parameters for the model.

    Return the latency and energy cost mappings and the network shape
    statistics.
    """
    # Load the hardware cost data.
    published_data = load_hw_data()
    latency, power = layer_costs(published_data)
    energy = dict_product(latency, power)

    # Load the network information.
    net_data = load_net_data()

    return latency, energy, net_data


def layer_kind(name):
    """Return a short string indicating the kind of the named layer.
    """
    if name.startswith('CONV'):
        return 'conv'
    elif name.startswith('FC'):
        return 'fc'
    assert False


def model(config_file):
    """Run the model for a configuration given in the specified file.
    """
    latency, energy, net_data = load_params()
    latency_ratios = scaling_ratios(net_data, latency)
    energy_ratios = scaling_ratios(net_data, energy)

    # Load the configuration we're modeling.
    layers = load_config(config_file)

    # Initialize accumulators.
    totals = {}
    for kind in LAYER_KINDS:
        totals[kind] = { 'energy': 0.0, 'latency': 0.0 }

    # Add the cost for each layer.
    for layer in layers:
        if isinstance(layer, LookupLayer):
            # Use the built-in numbers for this layer.
            kind = layer_kind(layer.layer)
            totals[kind]['energy'] += energy[layer]
            totals[kind]['latency'] += latency[layer]

        elif isinstance(layer, ScaleLayer):
            # Scale the average costs.
            energy = energy_ratios[layer.kind] * layer.macs
            latency = latency_ratios[layer.kind] * layer.macs
            totals[layer.kind]['energy'] += energy
            totals[layer.kind]['latency'] += latency

    # Grand totals.
    totals['total'] = {
        'energy': sum(totals[k]['energy'] for k in LAYER_KINDS),
        'latency': sum(totals[k]['latency'] for k in LAYER_KINDS),
    }

    return totals


def diagnose_scaled_cost(net_data, costs):
    """Get information for diagnosing FODLAM's scaling logic for a
    particular cost dimension.

    For the given cost mapping, return the cost per MAC of each layer
    for each model.
    """
    out = {}
    for net, layer_macs in net_data.items():
        net_costs = {}
        for layer, macs in layer_macs.items():
            cost = costs[net, layer]
            cost_per_mac = cost / macs
            net_costs[layer] = cost_per_mac
        out[net] = net_costs
    return out


def diagnose_scaling():
    """Get per-MAC costs for the latency and energy of each layer and
    overall averages.
    """
    latency, energy, net_data = load_params()
    return {
        'per_layer': {
            'latency': diagnose_scaled_cost(net_data, latency),
            'energy': diagnose_scaled_cost(net_data, energy),
        },
        'average': {
            'latency': scaling_ratios(net_data, latency),
            'energy': scaling_ratios(net_data, energy),
        },
    }


if __name__ == '__main__':
    if sys.argv[1:] and sys.argv[1] == '--diagnose':
        out = diagnose_scaling()
    else:
        out = model(sys.stdin)
    print(json.dumps(out, sort_keys=True, indent=2))
