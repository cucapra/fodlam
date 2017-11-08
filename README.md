First-Order Deep Learning Accelerator Model (FODLAM)
====================================================

This is a quick, easy model for the power and performance of modern hardware implementations of deep neural networks. It is based on published numbers from two papers:

* ["EIE: Efficient Inference Engine on Compressed Deep Neural Network."](https://arxiv.org/pdf/1602.01528.pdf)
  Song Han, Xingyu Liu, Huizi Mao, Jing Pu, Ardavan Pedram, Mark A. Horowitz, and William J. Dally.
  In ISCA 2016.
* ["Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks."](http://www.rle.mit.edu/eems/wp-content/uploads/2016/04/eyeriss_isca_2016.pdf)
  Yu-Hsin Chen, Tushar Krishna, Joel S. Emer, and Vivienne Sze.
  In J. Solid-State Circuits, January 2017.

EIE provides the fully-connected layers; Eyeriss provides the convolutional layers.


Data Extraction
---------------

I first extracted the raw data from the papers. The raw text files are in `raw/`.

* For EIE, I first used [Tabula][] to extract unstructured CSV data. I extracted tables II, IV, and V. (Table III was not referenced in the text; it just seems to characterize the benchmarks.)
* In the Eyeriss papers, the PDF does not have text embedded for the tables. I extracted images of tables III through VI and OCR'd them with [Tesseract][].

[tabula]: http://tabula.technology
[tesseract]: https://github.com/tesseract-ocr/tesseract
