#!/usr/bin/env python3
from __future__ import division, print_function

import os
import csv
import json
import sys

NETWORKS = ('VGG16', 'AlexNet')

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


def load_data():
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


def load_config(config_file, available_layers):
    """Load a neural network configuration from a file-like object.
    Return a set of enabled layers, which are pairs of strings (the
    network name and the layer name).

    Also, check that the configured layers are all available in the
    given set of layer IDs.
    """
    config_data = json.load(config_file)
    layers = set(tuple(l) for l in config_data['layers'])
    assert layers <= available_layers
    return layers


def model(config_file):
    """Run the model for a configuration given in the specified file.
    """
    # Load the layer data.
    published_data = load_data()
    latency, power = layer_costs(published_data)
    energy = dict_product(latency, power)

    # Load the configuration we're modeling.
    layers = load_config(config_file, set(energy))

    # Sum the latency and power for each enabled layer.
    total_latency = select_sum(layers, latency)
    total_energy = select_sum(layers, energy)

    return {
        'latency': total_latency,
        'energy': total_energy,
    }


if __name__ == '__main__':
    out = model(sys.stdin)
    print(json.dumps(out, sort_keys=True, indent=2))
