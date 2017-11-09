#!/usr/bin/env python3
from __future__ import division, print_function

import os
import csv
import json

DATA_DIR = 'data'
EIE_DATA = 'eie-layers.csv'
EYERISS_FILE = 'eyeriss-vgg16.csv'

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
    eie_vgg_latencies = {}
    with open(os.path.join(DATA_DIR, EIE_DATA)) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Layer'] == 'Actual Time':
                for k, v in row.items():
                    if k.startswith('VGG16'):
                        eie_vgg_latencies[k.split()[1]] = \
                            float(v) * EIE_TIME_UNIT

    # Load Eyeriss data (latency and energy).
    eyeriss_vgg = {
        'latency_total': {},
        'latency_proc': {},
        'power': {},
    }
    with open(os.path.join(DATA_DIR, EYERISS_FILE)) as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = row['Layer']
            if layer == 'Total':
                continue
            eyeriss_vgg['latency_total'][layer] = \
                float(row['Total Latency (ms)']) * EYERISS_TIME_UNIT
            eyeriss_vgg['latency_proc'][layer] = \
                float(row['Processing Latency (ms)']) * EYERISS_TIME_UNIT
            eyeriss_vgg['power'][layer] = \
                float(row['Power (mW)']) * EYERISS_POWER_UNIT

    return { 'eie': eie_vgg_latencies, 'eyeriss': eyeriss_vgg }


def layer_costs(published):
    """Get the latencies for *all* layers in VGG-16 by combining EIE and
    Eyeriss data.
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

    # Combine all the layers.
    latency = dict(eie_lat_scaled)
    latency.update(eyeriss_lat)
    power = { k: eie_power_scaled for k in eie_lat }  # Constant power.
    power.update(eyeriss_pow)

    return latency, power


if __name__ == '__main__':
    published_data = load_data()
    print(json.dumps(published_data, sort_keys=True, indent=2))
    latency, power = layer_costs(published_data)
    print(json.dumps(latency, sort_keys=True, indent=2))
    print(json.dumps(power, sort_keys=True, indent=2))
