import numpy as np
import tensorflow as tf
from IPython import embed
import cv2
from object_detection.utils import dataset_util
import os

with open("TrainImages.txt") as f:
	train_images = f.readlines()


def encode_image(path):
	image = cv2.imread(path)
	image = cv2.imencode('.jpg',image)[1].tostring()
	return image
	
writer = tf.python_io.TFRecordWriter("/home/deploy/shubham/pratice/classification/train.record")

for i in train_images:
	path_ = os.path.join("/home/deploy/shubham/pratice/classification/indoorCVPR_09/Images",i.strip())
	
	try:
		encoded_img = encode_image(path_)
	except:
		pass

	label = i.split("/")[0]
	label = tf.compat.as_bytes(label)

	tf_example = tf.train.Example(features = tf.train.Features(feature = {
		"image":dataset_util.bytes_feature(encoded_img),
		"label":dataset_util.bytes_feature(label)
		}))
	writer.write(tf_example.SerializeToString())
