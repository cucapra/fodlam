First-Order Deep Learning Accelerator Model (FODLAM)
====================================================

This is a quick, easy model for the power and performance of modern hardware implementations of deep neural networks. It is based on published numbers from two ISCA 2016 papers:

* ["EIE: Efficient Inference Engine on Compressed Deep Neural Network."](https://arxiv.org/pdf/1602.01528.pdf)
  Song Han, Xingyu Liu, Huizi Mao, Jing Pu, Ardavan Pedram, Mark A. Horowitz, and William J. Dally.
  In ISCA 2016.
* ["Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks."](http://www.rle.mit.edu/eems/wp-content/uploads/2016/04/eyeriss_isca_2016.pdf)
  Yu-Hsin Chen, Joel Emer, and Vivienne Sze.
  In ISCA 2016.

EIE provides the fully-connected layers; Eyeriss provides the convolutional layers.
