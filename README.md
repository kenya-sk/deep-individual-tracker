[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

# Deep Individual Tracker
"deep-individual-tracker" is a deep learning-based tracking method that takes into account the overlap of individuals to detect. This repository provides annotation tools, trackers, and monitoring tools.

## Modules
### Density Annotator
DensityAnnotator is a GUI-based annotation tool that creates a density map for an image. Not only image data but also video data can be handled, and it can be cut out into a frame at any timing and annotated in the same way as for images.

### Individual Detection
This repository performs individual detection in consideration of overlap by using CNN (Convolutional Neural Network) in each pixel.

### Individual Tracking
Tracking is performed based on the location of the individual detected by 'indevidual-detection'. When tracking, template matching is performed for the neighborhood of the detection point and correspondence is made between frames.

### Statistic Monitoring
The system provides a monitoring environment with the following statistical information added to the video.
- Histogram of X-axis and Y-axis position information of each individual tuna
- Time series plot of average cruising speed
- Time-series plot of the cumulative number of sudden accelerations
- Time series plot of the number of individuals detected by the machine learning model