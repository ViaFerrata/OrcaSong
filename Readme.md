## OrcaSong: Generating DL images based on KM3NeT data

[![alt text][image_1]][hyperlink_1] [![alt text][image_2]][hyperlink_2]

  [hyperlink_1]: https://git.km3net.de/ml/OrcaSong/pipelines
  [image_1]: https://git.km3net.de/ml/OrcaSong/badges/master/build.svg

  [hyperlink_2]: https://ml.pages.km3net.de/OrcaSong
  [image_2]: https://examples.pages.km3net.de/km3badges/docs-latest-brightgreen.svg
  

The documentation for OrcaSong can be found at https://ml.pages.km3net.de/OrcaSong!

OrcaSong is a part of the Deep Learning efforts of the neutrino telescope KM3NeT.  
Find more information about KM3NeT on http://www.km3net.org.

In this regard, OrcaSong is a project that produces KM3NeT event images based on the raw detector data.
This means that OrcaSong takes a datafile with (neutrino-) events and based on this data, it produces 2D/3D/4D 'images' (histograms).
Currently, only simulations with a hdf5 data format are supported as an input.

These event 'images' are required for some Deep Learning machine learning algorithms, e.g. Convolutional Neural Networks.
