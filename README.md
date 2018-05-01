# Single Shot MultiBox Detector in TensorFlow

This repository contains codes of the reimplementation of [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) in TensorFlow. If your goal is to reproduce the results in the original paper, please use the official [codes](https://github.com/weiliu89/caffe/tree/ssd).

There are already some TensorFlow based SSD reimplementation codes on GitHub, the main special features of this repo inlcudes:

- state of the art performance when training from VGG-16 pre-trained model (trained model will be released later).
- the model is trained using TensorFlow high level API [tf.estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator). Although TensorFlow provides many APIs, the Estimator API is highly recommended to yield scalable, high-performance models. 
- all codes were writen by TensorFlow ops (no numpy operation) to ensure the performance and portability.
- using ssd augmentation pipeline discribed in the original paper.
- PyTorch-like model definition using high-level [tf.layers](https://www.tensorflow.org/api_docs/python/tf/layers) API for better readability ^-^.
- high degree of modularity to ease futher development.

## ##
## Usage
- Download [Pascal VOC Dataset](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) and reorganize the directory as follows:
	```
	VOCROOT/
		   |->VOC2007/
		   |    |->Annotations/
		   |    |->ImageSets/
		   |    |->...
		   |->VOC2012/
		   |    |->Annotations/
		   |    |->ImageSets/ 
		   |    |->...
		   |->VOC2007TEST/
		   |    |->Annotations/
		   |    |->...
	```
	VOCROOT is your path of the Pascal VOC Dataset.
- Run the following script to generate TFRecords.
	```sh
	python dataset/convert_tfrecords.py --dataset_directory=VOCROOT --output_directory=./dataset/tfrecords
	```
- Download the pre-trained VGG-16 model from [here](https://drive.google.com/drive/folders/184srhbt8_uvLKeWW_Yo8Mc5wTyc0lJT7) and put them into one sub-directory named 'model'.
- Run the following script to start training:

	```sh
	python train_ssd.py 
	```
- Run the following script for evaluation and get mAP:

	```sh
	python eval_ssd.py 
	python voc_eval.py 
	```
	Note: you need first modify some directory in voc_eval.py.
- Run the following script for visualization:
	```sh
	python simple_ssd_demo.py
	```

All the codes was tested under TensorFlow 1.6, Python 3.5, Ubuntu 16.04 with CUDA 8.0. The training is now still in processing, and the performance will be reported after the first training.

This repo is just created recently, any contribution will be welcomed.

## ##
Apache License, Version 2.0