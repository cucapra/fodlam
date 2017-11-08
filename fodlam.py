import os
import csv

DATA_DIR = 'data'
EIE_DATA = 'eie-layers.csv'
EYERISS_FILE = 'eyeriss-vgg16.csv'

def load_data():
    """Load the published numbers from our data files.
    """
    eie_vgg_latencies = {}
    with open(os.path.join(DATA_DIR, EIE_DATA)) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Layer'] == 'Actual Time':
                for k, v in row.items():
                    if k.startswith('VGG16'):
                        eie_vgg_latencies[k] = v
    print(eie_vgg_latencies)

    with open(os.path.join(DATA_DIR, EYERISS_FILE)) as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(row)

if __name__ == '__main__':
    load_data()
