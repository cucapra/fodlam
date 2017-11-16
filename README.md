First-Order Deep Learning Accelerator Model (FODLAM)
====================================================

FODLAM is a quick, easy model for the power and performance of modern hardware implementations of deep neural networks. It is based on published numbers from two papers:

* ["EIE: Efficient Inference Engine on Compressed Deep Neural Network."](https://arxiv.org/pdf/1602.01528.pdf)
  Song Han, Xingyu Liu, Huizi Mao, Jing Pu, Ardavan Pedram, Mark A. Horowitz, and William J. Dally.
  In ISCA 2016.
* ["Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks."](http://www.rle.mit.edu/eems/wp-content/uploads/2016/04/eyeriss_isca_2016.pdf)
  Yu-Hsin Chen, Tushar Krishna, Joel S. Emer, and Vivienne Sze.
  In J. Solid-State Circuits, January 2017.

EIE provides the fully-connected layers; Eyeriss provides the convolutional layers. FODLAM only supports these two kinds of layers.


Running the Model
-----------------

FODLAM is a Python 3 program. It has no other dependencies.

To specify a DNN, create a JSON file containing two keys:

* Choose one of these two options to select a network to draw layers from:
    * `net`: A built-in network name, either `"VGG16"` or `"AlexNet"`. FODLAM will use precise published numbers.
    * `netfile`: The name of a JSON file in the `nets/` directory that describes any CNN. FODLAM will approximate layer costs using scaling.
* `layers`: A list of layer names to enable.

You can see examples in `config/`.

Run FODLAM by piping in a configuration file, like this:

    $ python3 fodlam.py < config/vgg16.json
    {
      "conv": {
        "energy": 1.0162585,
        "latency": 4.3094
      },
      "fc": {
        "energy": 9.157180384087789e-05,
        "latency": 7.438888888888888e-05
      },
      "total": {
        "energy": 1.016350071803841,
        "latency": 4.309474388888889
      }
    }

The results are printed as JSON to stdout. The output consists of the total energy in joules and total latency in seconds. The output includes the total for the entire network, just the convolutional layers, and just the fully-connected layers.

### Providing a Network

FODLAM ships with statistics for a few popular neural networks as JSON files under the `nets/` directory. This JSON file describes the total computational cost of each layer in the network.

To provide a new network specification, you need to produce a similar JSON file. FODLAM has a tool that can extract these statistics from Caffe models, but unlike FODLAM itself, this tool requires a working Caffe installation. (You can even use a funky hacked-up alternative versions of Caffe, such as [the one for Fast and Faster R-CNN][caffe-fast-rcnn].) See the Makefile in that directory for tips on how to extract a JSON statistics file from your network specification.

[caffe-fast-rcnn]: https://github.com/rbgirshick/caffe-fast-rcnn


How it Works
------------

The model just totals up the latency and energy for each layer in a given configuration. Because both of the source papers measure AlexNet and VGG-16, layers from those networks are supported directly. For other layers, FODLAM can scale the data from those networks.

### Process Normalization

Because Eyeriss and EIE were evaluated on different process technologies, we have to scale one of them to model a single ASIC. Specifically, Eyeriss is on TSMC 65nm and EIE is on TSMC 45nm; we normalize to 65nm. This works by multiplying EIE time by the scaling factor and multiplying the power by the square of the scaling factor---i.e., Dennard scaling, which is admittedly retro.

### Power

While the Eyeriss paper reports per-layer power, the EIE paper does not. Instead, this is how energy is computed (quoting from the paper):

> Energy is obtained by multiplying computation time and total measured power...

So the authors assume that power is constant across layers. FODLAM applies the same assumption to compute EIE layer energy.

### New Layers

To estimate the costs for new layer configurations not found in AlexNet or VGG-16, FODLAM can scale the numbers from those networks. Scaling works by getting the number of multiply--accumulate (MAC) operations required to compute each layer. We compute the average cost per MAC among layers of the same type and use that to estimate the cost of a new layer.

The assumption underlying this scaling technique is that the cost per MAC is close to constant across layers of varying shape. To validate this hypothesis, run FODLAM in diagnosis mode:

    $ python3 fodlam.py --diagnose

FODLAM will print out the energy and latency per MAC for each layer. Notice that the cost per MAC is different for convolutional and fully-connected layers, but it varies by less than an order of magnitude within each layer type.


Data Extraction
---------------

To make FODLAM, I extracted raw data from tables in the papers. The raw text files from this extraction are in `raw/`.

* For EIE, I first used [Tabula][] to extract unstructured CSV data. I extracted tables II, IV, and V. (Table III was not referenced in the text; it just seems to characterize the benchmarks.)
* In the Eyeriss journal paper, the PDF does not have text embedded for the tables. I extracted images of tables III through VI and OCR'd them with [Tesseract][]. There were a lot of errors.

I then cleaned up the relevant data by hand. The cleaned-up CSVs that FODLAM uses are in `data/`.

[tabula]: http://tabula.technology
[tesseract]: https://github.com/tesseract-ocr/tesseract


Credits
-------

This is a research artifact from [Capra][] at Cornell. The license is [MIT][].

[capra]: https://capra.cs.cornell.edu
[mit]: https://opensource.org/licenses/MIT
