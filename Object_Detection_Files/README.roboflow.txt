
Traffic Cones - v2 augmentated version
==============================

This dataset was exported via roboflow.com on October 4, 2022 at 8:52 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

It includes 1554 images.
Cone are annotated in YOLO v3 (Keras) format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Random rotation of between -22 and +22 degrees
* Random brigthness adjustment of between -30 and +30 percent


