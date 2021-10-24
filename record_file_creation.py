import numpy as np
import tensorflow as tf
from IPython import embed
import cv2
from object_detection.utils import dataset_util
import os

lbl = ["pantry","laundromat","dentaloffice","classroom","laboratorywet","hospitalroom","nursery","bookstore","gym","kindergarden","hairsalon","operating_room","casino","office","corridor","greenhouse","museum","concert_hall","elevator","warehouse","restaurant","waitingroom","locker_room","toystore","bakery","stairscase","bathroom","bar","trainstation","children_room","jewelleryshop","videostore","airport_inside","church_inside","winecellar","buffet","garage","inside_bus","shoeshop","livingroom","bedroom","poolinside","closet","dining_room","meeting_room","bowling","studiomusic","clothingstore","grocerystore","cloister","restaurant_kitchen","kitchen","gameroom","subway","deli","artstudio","fastfood_restaurant","florist","auditorium","computerroom","mall","tv_studio","movietheater","lobby","library","inside_subway","prisoncell"]
lbl_ = {}

for i,j in enumerate(lbl, start=0):
	lbl_[j] = i


print("label",lbl_)


# tf.compat.v1.enabl
with open("TrainImages.txt") as f:
	train_images = f.readlines()

def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def encode_image(path):
	# image = cv2.imread(path)
	image = tf.keras.utils.load_img(path)
	return image
	
writer = tf.io.TFRecordWriter("/home/deploy/shubham/pratice/classification/train.record")

for i in train_images:
	path_ = os.path.join("/home/deploy/shubham/pratice/classification/indoorCVPR_09/Images",i.strip())
	
	# try:
	encoded_img = encode_image(path_)
	# except:
	# 	print(path_)
	# 	continue

	label = lbl_[i.split("/")[0]]
	# label = tf.compat.as_bytes(label)

	tf_example = tf.train.Example(features = tf.train.Features(feature = {
		"image":image_feature(encoded_img),
		"label":int64_feature(label)
		}))
	writer.write(tf_example.SerializeToString())